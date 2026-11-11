# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-alpha-compositing: a sticker laid over a background with its alpha channel, and the alpha
of a photo in front of a green screen, from the hue and the saturation of every pixel. And the drawing the lesson
uses: four pieces of fruit with their shadows, as a sticker with an alpha channel, and the same sticker in front of a
green screen. The choice about the axis of the alpha channel and the open question about the edges are in the lesson
itself."""
import functools
import re

import numpy as np

from everything_ai.check import CodeBox, FunctionBox, Mistake
from everything_ai import yours

STICKER = "data/photos/fruit-sticker.png"   # 400 rows by 600 columns, RGBA: the fruit and their shadows (draw_sticker)
SCREEN = "data/photos/green-screen.png"     # the same sticker in front of a green screen, RGB (draw_green_screen)
COFFEE = "data/photos/coffee.png"           # the photo of cv-image-files: 400 rows by 600 columns, RGB
BOUNDS = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40]   # the stops of the slider: where the green of the key starts

# the fruit: what it is, row, column, radius, hue, saturation
FRUIT = [("orange", 60, 530, 38, 0.075, 0.90), ("lime", 245, 556, 30, 0.270, 0.75),
         ("lemon", 336, 543, 36, 0.140, 0.80), ("apple", 344, 54, 38, 0.000, 0.85)]
SCREEN_HUE = 0.39                           # a bluish green, as real green screens


def draw_sticker() -> np.ndarray:
    """The sticker of the lesson: four pieces of fruit, each a ball with light from the upper right, a darker side and
    a white shine, with a soft shadow on the lower left. Around them everything is transparent. Every edge is soft:
    a pixel on the edge of a ball gets the part of it that the ball covers as its alpha, and the shadow is black with
    an alpha of at most 0.45. Gives an RGBA image of uint8, 400 rows by 600 columns."""
    from matplotlib.colors import hsv_to_rgb
    rows, cols = np.mgrid[0:400, 0:600].astype(float)
    rgb = np.zeros((400, 600, 3))
    alpha = np.zeros((400, 600))
    for _, r0, c0, radius, _, _ in FRUIT:
        ey, ex = (rows - (r0 + 0.80 * radius)) / (0.42 * radius), (cols - (c0 - 0.30 * radius)) / (1.05 * radius)
        shadow = 0.45 * np.clip((1 - np.sqrt(ey ** 2 + ex ** 2)) / 0.45, 0, 1)
        rgb = rgb * (alpha / np.maximum(alpha + (1 - alpha) * shadow, 1e-9))[..., None]   # a shadow is black
        alpha = alpha + (1 - alpha) * shadow
    for _, r0, c0, radius, hue, sat in FRUIT:
        cover = np.clip(radius - np.hypot(rows - r0, cols - c0) + 0.5, 0, 1)
        dy, dx = (rows - r0) / radius, (cols - c0) / radius
        light = np.clip(0.45 * dx - 0.55 * dy + 0.70 * np.sqrt(np.clip(1 - dx ** 2 - dy ** 2, 0, 1)), 0, 1)
        shine = np.clip(1 - np.hypot(dx - 0.38, dy + 0.42) / 0.16, 0, 1)
        value = 0.30 + 0.62 * light
        fruit = hsv_to_rgb(np.stack([np.full(rows.shape, hue), sat * (1 - shine), value + (1 - value) * shine], axis=-1))
        new = cover + (1 - cover) * alpha
        rgb = (cover[..., None] * fruit + ((1 - cover) * alpha)[..., None] * rgb) / np.maximum(new, 1e-9)[..., None]
        alpha = new
    out = np.zeros((400, 600, 4), dtype=np.uint8)
    out[..., :3] = np.round(rgb * 255)
    out[..., 3] = np.round(alpha * 255)
    out[out[..., 3] == 0, :3] = 0
    return out


def _screen() -> np.ndarray:
    """The green screen behind the fruit, as floats from 0 to 255: a bluish green with soft folds, darker towards the
    corners, where the lamps give less light."""
    from matplotlib.colors import hsv_to_rgb
    rows, cols = np.mgrid[0:400, 0:600].astype(float)
    fold = np.sin(cols / 37 + rows / 90) * np.sin(rows / 53 - cols / 140)
    far = ((rows - 170) / 260) ** 2 + ((cols - 280) / 420) ** 2
    return hsv_to_rgb(np.stack([SCREEN_HUE + 0.008 * fold, 0.80 + 0.05 * fold, 0.80 - 0.22 * far + 0.04 * fold], axis=-1)) * 255


def draw_green_screen() -> np.ndarray:
    """The photo in front of the green screen: the sticker laid over the screen, as the lesson lays one image over
    another. Gives an RGB image of uint8, 400 rows by 600 columns."""
    sticker = draw_sticker()
    a = sticker[:, :, 3:] / 255
    return rounded(a * sticker[:, :, :3] + (1 - a) * _screen())


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


def rounded(out) -> np.ndarray:
    """Back to whole numbers from 0 to 255, rounded, as the recipe does it: `(out + 0.5).astype(np.uint8)`."""
    return (np.asarray(out) + 0.5).astype(np.uint8)


def over(sticker, back) -> np.ndarray:
    """The course's alpha compositing: the sticker over the background, as floats."""
    a = sticker[:, :, 3:] / 255
    return a * sticker[:, :, :3] + (1 - a) * back


@functools.lru_cache(maxsize=1)
def true_alpha() -> np.ndarray:
    """The alpha of the sticker, from 0 to 1, shape (400, 600, 1): what a key of the green screen should find."""
    return read(STICKER)[:, :, 3:] / 255


@functools.lru_cache(maxsize=1)
def _screen_hsv() -> np.ndarray:
    from matplotlib.colors import rgb_to_hsv
    return rgb_to_hsv(read(SCREEN) / 255)


def key(low: float, high: float = 0.45, saturation: float = 0.5) -> np.ndarray:
    """The alpha of the green-screen photo from its hue and saturation: 0 where the screen is, 1 elsewhere."""
    hsv = _screen_hsv()
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    screen = (h > low) & (h < high) & (s > saturation)
    return (1 - screen).reshape(400, 600, 1)


@functools.lru_cache(maxsize=None)
def keyed(low: float) -> np.ndarray:
    """The green-screen photo laid over the coffee photo, with the alpha of `key(low)`."""
    a = key(low)
    return rounded(a * read(SCREEN) + (1 - a) * read(COFFEE))


@functools.lru_cache(maxsize=1)
def fruit_parts() -> dict[str, np.ndarray]:
    """For each piece of fruit, its pixels away from the edge, shape (400, 600)."""
    rows, cols = np.mgrid[0:400, 0:600]
    return {name: (rows - r0) ** 2 + (cols - c0) ** 2 < (radius - 1) ** 2 for name, r0, c0, radius, _, _ in FRUIT}


def key_outcome(alpha) -> dict:
    """What a key makes of the photo: how many pixels of each fruit it takes for screen, how many pixels of the
    screen it keeps, and in how many pixels it differs from the sticker's alpha, both cut at a half."""
    alpha = np.asarray(alpha, dtype=float).reshape(400, 600)
    true = true_alpha()[:, :, 0]
    lost = {name: int((alpha[part] < 0.5).sum()) for name, part in fruit_parts().items()}
    return {"lost": lost, "left": int(((true == 0) & (alpha >= 0.5)).sum()),
            "differ": int(((alpha >= 0.5) != (true > 0.5)).sum())}


def _panel(ax, img, title, **kw):
    ax.imshow(img, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def alpha_picture(labels: dict[str, str]) -> str:
    """The alpha channel of the sticker as a gray image, and the sticker laid over the coffee photo."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    sticker = read(STICKER)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.04)
    _panel(axes[0], sticker[:, :, 3], labels["alpha"], cmap="gray", vmin=0, vmax=255)
    _panel(axes[1], rounded(over(sticker, read(COFFEE))), labels["over"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def key_frame(low: float, labels: dict[str, str]) -> str:
    """For the slider: the alpha of the key whose green starts at `low`, and the green-screen photo laid over the
    coffee photo with it, drawn at half size so that all stops of the slider stay light to send."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from everything_ai import plots
    low = round(float(low), 2)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.04)
    _panel(axes[0], key(low)[:, :, 0], labels["alpha"], cmap="gray", vmin=0, vmax=1)
    _panel(axes[1], np.array(Image.fromarray(keyed(low)).resize((300, 200), Image.BILINEAR)), labels["over"])
    html = plots.render(fig, alt=labels["alt"].format(low=low))
    plt.close(fig)
    return html


def edge_picture(labels: dict[str, str]) -> str:
    """The orange up close: on the left with the key of the lesson, on the right with the sticker's own alpha."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    part = (slice(14, 108), slice(478, 582))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.04)
    for ax, img, title in zip(axes, (keyed(0.3)[part], rounded(over(read(STICKER), read(COFFEE)))[part]),
                              (labels["key"], labels["sticker"])):
        _panel(ax, img, title, interpolation="nearest")
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


REQUIRED = ["axis", "over", "key", "edges"]

# a `...` the learner has not replaced yet; `...` right after `[` or right before `,` or `]` picks out an axis, as in
# `sticker[..., 3:]`, and is right
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*(?<!\[)\.\.\.(?!\s*[,\]])", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}

_SWAP = {"en": "`:3` and `3:` are swapped somewhere. `sticker[:, :, :3]` takes the numbers before place 3, the colors; `sticker[:, :, 3:]` takes them from place 3 on, the alpha. `a` comes from the alpha, and the front is the colors.",
         "nl": "`:3` en `3:` zijn ergens omgewisseld. `sticker[:, :, :3]` neemt de getallen vóór plek 3, de kleuren; `sticker[:, :, 3:]` neemt ze vanaf plek 3, de alpha. `a` komt uit de alpha, en de voorkant zijn de kleuren."}


def _m(key_, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key_, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


class _Img(np.ndarray):
    """An image of a case, shown in the check's messages as the code that makes it."""

    def __repr__(self):
        if self.dtype == np.uint8:
            return f"np.array({np.asarray(self).tolist()}, dtype=np.uint8)"
        return f"np.array({np.asarray(self).tolist()})"


def _img(rows) -> np.ndarray:
    return np.array(rows, dtype=np.uint8).view(_Img)


def _shown(v):
    """What the learner's function gave, the way a cell shows it, rounded to two decimals."""
    if isinstance(v, np.ndarray):
        arr = np.asarray(v)
        if arr.dtype.kind == "f":
            arr = np.round(arr, 2)
            if np.all(arr == np.round(arr)):
                return str(arr.astype(int).tolist())
        return str(arr.tolist())
    return repr(v)


# Exercise 2: the sticker over the background
_OVER_HEAD = "def over(sticker, back):\n"
_OVER_START = _OVER_HEAD + "    a = ...\n    return ...\n"


def _over(a: str = "sticker[:, :, 3:] / 255", back: str = "a * sticker[:, :, :3] + (1 - a) * back") -> str:
    """The course's function with another alpha, or another mix."""
    return _OVER_HEAD + f"    a = {a}\n    return {back}\n"


_OVER = _over()

# The example of exercise 2: an orange pixel at alpha 0.2 over a blue one, and a shadow at alpha 0.4 over the table
EXAMPLE_STICKER = _img([[[240, 120, 0, 51], [0, 0, 0, 102]]])
EXAMPLE_BACK = _img([[[0, 0, 240], [180, 120, 80]]])
_EDGES = (_img([[[250, 10, 0, 255], [10, 250, 0, 0]], [[0, 0, 250, 153], [250, 250, 250, 204]]]),
          _img([[[5, 5, 5], [100, 200, 30]], [[255, 255, 0], [0, 0, 0]]]))
_DARK = (_img([[[20, 40, 60, 128], [0, 0, 0, 255]], [[200, 100, 50, 77], [90, 90, 90, 26]]]),
         _img([[[220, 200, 180], [255, 255, 255]], [[10, 20, 30], [250, 240, 230]]]))

# Exercise 3: the alpha of the green-screen photo
_KEY_HEAD = ("import numpy as np\n"
             "from PIL import Image\n"
             "from matplotlib.colors import rgb_to_hsv\n\n"
             "photo = np.array(Image.open(\"data/photos/green-screen.png\"))\n"
             "hsv = rgb_to_hsv(photo / 255)\n"
             "h, s = hsv[:, :, 0:1], hsv[:, :, 1:2]\n")
_KEY_START = _KEY_HEAD + "screen = ...\nalpha = ...\nalpha\n"


def _key(screen: str = "(h > 0.3) & (h < 0.45) & (s > 0.5)", alpha: str = "1 - screen",
         last: str = "alpha") -> str:
    """The box of exercise 3 with these lines for `screen` and `alpha`."""
    return _KEY_HEAD + f"screen = {screen}\nalpha = {alpha}\n{last}\n"


_KEY = _key()


def _key_right(got, want) -> bool:
    """A key is right when its alpha has the shape of the sticker's, lies from 0 to 1, keeps every fruit and drops
    the screen: at a cut of a half, it differs from the sticker's alpha in at most 1 pixel in 500."""
    if not isinstance(got, np.ndarray) or got.shape != (400, 600, 1) or got.dtype.kind not in "biuf":
        return False
    if got.min() < 0 or got.max() > 1:
        return False
    outcome = key_outcome(got)
    return outcome["differ"] <= 480 and not any(outcome["lost"].values()) and outcome["left"] == 0


def _key_shape(*shape):
    return lambda got, text: isinstance(got, np.ndarray) and got.shape == shape


def _key_ok(got) -> bool:
    return isinstance(got, np.ndarray) and got.shape == (400, 600, 1) and got.dtype.kind in "biuf"


def _lost(name):
    return lambda got, text: _key_ok(got) and key_outcome(got)["lost"][name] > 100


EXERCISES = {
    "over": FunctionBox(
        name="over",
        picture=yours.image(call=0),
        function="over",
        start=_OVER_START,
        solution=_OVER,
        visible=[(EXAMPLE_STICKER, EXAMPLE_BACK)],
        hidden=[_EDGES, _DARK],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("rgba", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e) and ",4)" in str(e).replace(" ", ""),
                    en="With `{call}`, your function stops with an error: it mixes all four numbers of the sticker, alpha too, with the three of the background. Take the colors alone, `sticker[:, :, :3]`, for the front.",
                    nl="Met `{call}` stopt je functie met een fout: hij mengt alle vier de getallen van de sticker, alpha ook, met de drie van de achtergrond. Neem alleen de kleuren, `sticker[:, :, :3]`, voor de voorkant."),
            Mistake("front", error=lambda e: isinstance(e, NameError) and "'front'" in str(e),
                    en="With `{call}`, your function stops with an error: Python does not know `front` inside your function. The recipe makes it first, with `front = sticker[:, :, :3]`. Write that line in your function too, or write `sticker[:, :, :3]` where `front` is.",
                    nl="Met `{call}` stopt je functie met een fout: Python kent `front` niet binnen je functie. Het recept maakt hem eerst, met `front = sticker[:, :, :3]`. Schrijf die regel ook in je functie, of schrijf `sticker[:, :, :3]` waar `front` staat."),
            Mistake("axis", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="With `{call}`, your function stops with an error: `sticker[:, :, 3]` has one axis less than the colors, and lined up on the right, the two do not fit. Keep the axis with `sticker[:, :, 3:]`: one number per pixel, as a column of length 1 that NumPy stretches over the three colors.",
                    nl="Met `{call}` stopt je functie met een fout: `sticker[:, :, 3]` heeft één as minder dan de kleuren, en rechts uitgelijnd passen de twee niet. Houd de as met `sticker[:, :, 3:]`: één getal per pixel, als een kolom van lengte 1 die NumPy over de drie kleuren uitrekt."),
            _m("no-255", "`a` is still a number from 0 to 255, in uint8: `1 - a` wraps around, and `a * sticker[:, :, :3]` too. Divide the alpha channel by 255 first, so `a` runs from 0 to 1.",
               "`a` is nog een getal van 0 tot en met 255, in uint8: `1 - a` loopt rond, en `a * sticker[:, :, :3]` ook. Deel het alpha channel eerst door 255, zodat `a` van 0 tot en met 1 loopt.",
               wrong=_over(a="sticker[:, :, 3:]")),
            _m("floor", "`//` divides and drops what comes after the point: 51 // 255 is 0, so `a` is 0 for every alpha below 255. Divide with `/`, which keeps it: 51 / 255 is 0.2.",
               "`//` deelt en laat weg wat na de komma komt: 51 // 255 is 0, dus `a` is 0 voor elke alpha onder 255. Deel met `/`, dat houdt het: 51 / 255 is 0,2.",
               wrong=_over(a="sticker[:, :, 3:] // 255")),
            *[_m(key_, _SWAP["en"], _SWAP["nl"], wrong=_over(**kw)) for key_, kw in (
                ("swap", {"a": "sticker[:, :, :3] / 255", "back": "a * sticker[:, :, 3:] + (1 - a) * back"}),
                ("swap-alpha", {"a": "sticker[:, :, :3] / 255"}),
                ("swap-front", {"back": "a * sticker[:, :, 3:] + (1 - a) * back"}))],
            _m("flip", "The weights are the wrong way around: where the sticker covers, the background shows. `a` is how much the sticker covers, so the sticker gets `a`, and the background `1 - a`.",
               "De gewichten staan omgekeerd: waar de sticker bedekt, zie je de achtergrond. `a` is hoeveel de sticker bedekt, dus de sticker krijgt `a`, en de achtergrond `1 - a`.",
               wrong=_over(back="(1 - a) * sticker[:, :, :3] + a * back")),
            _m("whole-back", "The background counts in full, also where the sticker covers it, so the numbers go above 255. Give the background `1 - a`: what the sticker leaves of it.",
               "De achtergrond telt helemaal mee, ook waar de sticker hem bedekt, dus de getallen komen boven 255. Geef de achtergrond `1 - a`: wat de sticker ervan overlaat.",
               wrong=_over(back="a * sticker[:, :, :3] + back")),
            _m("parentheses", "Python multiplies before it subtracts: `1 - a * back` takes `a * back` away from 1. Put `1 - a` in parentheses: `(1 - a) * back`.",
               "Python vermenigvuldigt voordat het aftrekt: `1 - a * back` haalt `a * back` van 1 af. Zet `1 - a` tussen haakjes: `(1 - a) * back`.",
               wrong=_over(back="a * sticker[:, :, :3] + 1 - a * back")),
            _m("front-only", "Only the sticker counts: where it does not cover, everything turns black. Add what the sticker leaves of the background: `(1 - a) * back`.",
               "Alleen de sticker telt: waar hij niet bedekt, wordt alles zwart. Tel erbij op wat de sticker van de achtergrond overlaat: `(1 - a) * back`.",
               wrong=_over(back="a * sticker[:, :, :3]")),
            _m("wrap", "`sticker[:, :, :3] - back` is uint8 and cannot go below 0: 0 - 240 wraps around to 16, as in [From light to a file of numbers](/course/open/cv-image-files?lang=en). Use the form with `a` and `1 - a`, which only adds.",
               "`sticker[:, :, :3] - back` is uint8 en kan niet onder 0: 0 - 240 loopt rond naar 16, zoals in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl). Gebruik de vorm met `a` en `1 - a`, die alleen optelt.",
               wrong=_over(back="back + a * (sticker[:, :, :3] - back)")),
            _m("rounded", "Your function rounds already. Give back the mix as floats: the recipe rounds once, at the end, for the whole image.",
               "Je functie rondt al af. Geef de mix terug als floats: het recept rondt één keer af, aan het eind, voor het hele beeld.",
               matches=lambda got, text: isinstance(got, np.ndarray) and (got.dtype.kind in "iu" or (
                   got.dtype.kind == "f" and bool(np.all(got == np.round(got)))))),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: `a` is the alpha channel of the sticker, from 0 to 1, with its axis kept; the function gives back the colors of the sticker times `a`, plus the background times `1 - a`.",
                      "nl": "Vervang elke `...`: `a` is het alpha channel van de sticker, van 0 tot en met 1, met zijn as erbij; de functie geeft de kleuren van de sticker keer `a` terug, plus de achtergrond keer `1 - a`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a sticker and a background, and the `def` line of your function has a different number of names between its parentheses. Keep `(sticker, back)`.",
                          "nl": "De check roept `{call}` aan, met een sticker en een achtergrond, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(sticker, back)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the first pixel by hand: alpha 51 gives `a = 0.2`, so red is 0.2 × 240 + 0.8 × 0 = 48. Which step of your function gives something else?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de eerste pixel met de hand uit: alpha 51 geeft `a = 0.2`, dus rood is 0,2 × 240 + 0,8 × 0 = 48. Welke stap van je functie geeft iets anders?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a pixel with alpha 0 or 255, and where the sticker is darker than the background?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor een pixel met alpha 0 of 255, en waar de sticker donkerder is dan de achtergrond?"},
            "correct": {"en": "Your function is right in all {n} cases. The orange pixel at alpha 0.2 gives 48, 24 and 192: mostly the blue behind it. The shadow at alpha 0.4 gives 108, 72 and 48: the table, 0.6 times as bright.",
                        "nl": "Je functie klopt in alle {n} gevallen. De oranje pixel met alpha 0,2 geeft 48, 24 en 192: vooral het blauw erachter. De schaduw met alpha 0,4 geeft 108, 72 en 48: de tafel, 0,6 keer zo helder."},
        },
    ),
    "key": CodeBox(
        name="key",
        picture=yours.image(show=("photo",)),
        start=_KEY_START,
        solution=_KEY,
        compare=_key_right,
        first=[_DOTS],
        mistakes=[
            Mistake("and", error=lambda e: isinstance(e, ValueError) and "truth value of an array" in str(e),
                    en="`and`, `or` and `a < h < b` want one True or False, and an array holds one per pixel. Join masks with `&`, each comparison in parentheses: `(h > 0.3) & (h < 0.45)`, as in [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en).",
                    nl="`and`, `or` en `a < h < b` willen één True of False, en een array heeft er één per pixel. Verbind maskers met `&`, elke vergelijking tussen haakjes: `(h > 0.3) & (h < 0.45)`, zoals in [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl)."),
            Mistake("parentheses", error=lambda e: isinstance(e, TypeError) and "bitwise_and" in str(e),
                    en="Python does `&` before `<` and `>`, so `h > 0.3 & h < 0.45` first tries `0.3 & h`. Put every comparison in its own parentheses: `(h > 0.3) & (h < 0.45)`.",
                    nl="Python doet `&` vóór `<` en `>`, dus `h > 0.3 & h < 0.45` probeert eerst `0.3 & h`. Zet elke vergelijking tussen haakjes: `(h > 0.3) & (h < 0.45)`."),
            Mistake("flat", matches=_key_shape(400, 600),
                    en="Your alpha has shape `(400, 600)`: one axis less than the colors of the photo, `(400, 600, 3)`, so NumPy cannot mix the two. Keep the axis, as the box does: `h = hsv[:, :, 0:1]` and `s = hsv[:, :, 1:2]` have shape `(400, 600, 1)`, and so has a mask made of them.",
                    nl="Je alpha heeft shape `(400, 600)`: één as minder dan de kleuren van de foto, `(400, 600, 3)`, dus NumPy kan de twee niet mengen. Houd de as, zoals het vak doet: `h = hsv[:, :, 0:1]` en `s = hsv[:, :, 1:2]` hebben shape `(400, 600, 1)`, en een masker daarvan ook."),
            Mistake("shape-last", matches=lambda got, text: isinstance(got, tuple) and tuple(got) == (400, 600, 1),
                    en="The shape is right. The check wants the alpha itself, so end the box with the line `alpha`, as it was.",
                    nl="De shape klopt. De check wil de alpha zelf, dus eindig het vak met de regel `alpha`, zoals hij was."),
            Mistake("scale", matches=lambda got, text: _key_ok(got) and got.max() > 1,
                    en="Your alpha goes above 1. In the mix, alpha runs from 0, the screen, to 1, the fruit: `1 - screen` gives exactly that.",
                    nl="Je alpha komt boven 1. In de mix loopt alpha van 0, het scherm, tot en met 1, het fruit: `1 - screen` geeft precies dat."),
            Mistake("inverted", matches=lambda got, text: _key_ok(got) and key_outcome(got)["left"] > 100_000
                    and sum(key_outcome(got)["lost"].values()) > 7_000,
                    en="Your alpha is the wrong way around: 1 for the screen and 0 for the fruit, so the fruit vanishes and the screen stays. Alpha is how much of the photo you keep: `1 - screen`.",
                    nl="Je alpha staat omgekeerd: 1 voor het scherm en 0 voor het fruit, dus het fruit verdwijnt en het scherm blijft. Alpha is hoeveel je van de foto houdt: `1 - screen`."),
            Mistake("orange", matches=_lost("orange"),
                    en="Your key takes the orange for screen: its hue is 0.075, and your rule lets the screen start below that. The screen's green lies from 0.38 to 0.40.",
                    nl="Je key houdt de sinaasappel voor scherm: zijn kleurtoon is 0,075, en je regel laat het scherm daaronder beginnen. Het groen van het scherm ligt van 0,38 tot 0,40."),
            Mistake("lemon", matches=_lost("lemon"),
                    en="Your key takes the lemon for screen: its yellow has hue 0.14, inside your range. The screen's green lies from 0.38 to 0.40: start the range higher.",
                    nl="Je key houdt de citroen voor scherm: zijn geel heeft kleurtoon 0,14, binnen je bereik. Het groen van het scherm ligt van 0,38 tot 0,40: begin het bereik hoger."),
            Mistake("lime", matches=_lost("lime"),
                    en="Your key takes the lime for screen. The lime is green too, but yellowish green: its hue is 0.27, inside your range. The rule for green in [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en) was made to find the lime; here the lime must stay. The screen's green lies from 0.38 to 0.40: start the range above 0.27.",
                    nl="Je key houdt de limoen voor scherm. De limoen is ook groen, maar geelgroen: zijn kleurtoon is 0,27, binnen je bereik. De regel voor groen in [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) was gemaakt om de limoen te vinden; hier moet de limoen blijven. Het groen van het scherm ligt van 0,38 tot 0,40: begin het bereik boven 0,27."),
            Mistake("screen-left", matches=lambda got, text: _key_ok(got) and key_outcome(got)["left"] > 0,
                    en="Part of the screen stays: your rule misses some of its green. The screen's hue lies from 0.38 to 0.40, and its saturation from just under 0.75 to 0.85. Let your range hold all of that, with room to spare: the slider counted every saturation above 0.5.",
                    nl="Een deel van het scherm blijft staan: je regel mist een deel van zijn groen. De kleurtoon van het scherm ligt van 0,38 tot 0,40, en de verzadiging van net onder 0,75 tot 0,85. Laat je bereik dat allemaal bevatten, met ruimte over: de slider telde elke verzadiging boven 0,5."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: `screen` is True where the hue lies in the screen's green and the saturation is strong, with `&` between the comparisons; `alpha` is `1 - screen`: 0 on the screen, 1 elsewhere.",
                      "nl": "Vervang de twee `...`: `screen` is True waar de kleurtoon in het groen van het scherm ligt en de verzadiging sterk is, met `&` tussen de vergelijkingen; `alpha` is `1 - screen`: 0 op het scherm, 1 overal anders."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box should be `alpha`, so the check sees it. Put it back.",
                         "nl": "De laatste regel van het vak moet `alpha` zijn, zodat de check hem ziet. Zet hem terug."},
            "wrong": {"en": "That is not yet the alpha of the photo. `alpha` should be an array of shape `(400, 600, 1)`, 0 where the screen is and 1 where the fruit is.",
                      "nl": "Dat is nog niet de alpha van de foto. `alpha` moet een array zijn met shape `(400, 600, 1)`, 0 waar het scherm is en 1 waar het fruit is."},
            "correct": {"en": "Right: your key keeps all four pieces of fruit and drops all of the screen. Next to the sticker's own alpha, your key is 1 where the sticker covers more than half and 0 where it covers less, in all but a few hundred of the 240,000 pixels. Those few hundred lie on the edges of the fruit.",
                        "nl": "Goed: je key houdt alle vier de stukken fruit en laat het hele scherm weg. Naast de eigen alpha van de sticker is je key 1 waar de sticker meer dan de helft bedekt en 0 waar hij minder bedekt, op een paar honderd van de 240.000 pixels na. Die paar honderd liggen op de randen van het fruit."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("over", _OVER_START, "empty", None),
    ("over", _OVER, "correct", None),
    ("over", _over(back="sticker[:, :, :3] * a + back * (1 - a)"), "correct", None),
    ("over", _over(a="sticker[:, :, 3:4] / 255.0"), "correct", None),
    ("over", _over(back="a * sticker[:, :, :3] + (1 - a) * back\n    print(a)"), "correct", None),
    ("over", _over(a="sticker[:, :, 3] / 255"), "error", "axis"),
    ("over", _over(back="a * sticker + (1 - a) * back"), "error", "rgba"),
    ("over", _over(a="sticker[:, :, 3:]"), "wrong", "no-255"),
    ("over", _over(back="(1 - a) * sticker[:, :, :3] + a * back"), "wrong", "flip"),
    ("over", _over(back="a * sticker[:, :, :3] + back"), "wrong", "whole-back"),
    ("over", _over(back="a * sticker[:, :, :3] + 1 - a * back"), "wrong", "parentheses"),
    ("over", _over(back="a * sticker[:, :, :3]"), "wrong", "front-only"),
    ("over", _over(back="back + a * (sticker[:, :, :3] - back)"), "wrong", "wrap"),
    ("over", _over(a="sticker[..., 3:] / 255", back="a * sticker[..., :3] + (1 - a) * back"), "correct", None),
    ("over", _over(a="sticker[:, :, 3:] // 255"), "wrong", "floor"),
    ("over", _over(a="sticker[:, :, :3] / 255", back="a * sticker[:, :, 3:] + (1 - a) * back"), "wrong", "swap"),
    ("over", _over(a="sticker[:, :, :3] / 255"), "wrong", "swap-alpha"),
    ("over", _over(back="a * sticker[:, :, 3:] + (1 - a) * back"), "wrong", "swap-front"),
    ("over", _over(back="a * front + (1 - a) * back"), "error", "front"),
    ("over", _over(back="np.round(a * sticker[:, :, :3] + (1 - a) * back)").replace(
        "def over", "import numpy as np\n\n\ndef over"), "wrong", "rounded"),
    ("over", _over(back="(a * sticker[:, :, :3] + (1 - a) * back + 0.5).astype(np.uint8)").replace(
        "def over", "import numpy as np\n\n\ndef over"), "wrong", "rounded"),
    ("over", _over(back="...").replace("    return ...\n", "    print(a * sticker[:, :, :3] + (1 - a) * back)\n"), "wrong", "prints"),
    ("over", _OVER_START.replace("    return ...\n", "    return a * sticker[:, :, :3] + (1 - a) * back\n"), "wrong", "dots-left"),
    ("key", _KEY_START, "empty", None),
    ("key", _KEY, "correct", None),
    ("key", _key("(h > 0.35) & (h < 0.5) & (s > 0.4)"), "correct", None),
    ("key", _key("(h > 0.3) & (h < 0.45)"), "correct", None),
    ("key", _key(alpha="~screen"), "correct", None),
    ("key", _key(alpha="1.0 - screen"), "correct", None),
    ("key", _key(alpha="(1 - screen).reshape(400, 600, 1)"), "correct", None),
    ("key", _key("h > 0.3 and h < 0.45"), "error", "and"),
    ("key", _key("0.3 < h < 0.45"), "error", "and"),
    ("key", _key("h > 0.3 & h < 0.45"), "error", "parentheses"),
    ("key", _key(alpha="(1 - screen)[:, :, 0]"), "wrong", "flat"),
    ("key", _KEY_HEAD + "h, s = hsv[:, :, 0], hsv[:, :, 1]\nscreen = (h > 0.3) & (h < 0.45) & (s > 0.5)\nalpha = 1 - screen\nalpha\n",
     "wrong", "flat"),
    ("key", _key(alpha="255 * (1 - screen)"), "wrong", "scale"),
    ("key", _key(alpha="screen"), "wrong", "inverted"),
    ("key", _key("(h > 0.05) & (h < 0.45) & (s > 0.5)"), "wrong", "orange"),
    ("key", _key("(h > 0.1) & (h < 0.45) & (s > 0.5)"), "wrong", "lemon"),
    ("key", _key("(h > 0.2) & (h < 0.45) & (s > 0.5)"), "wrong", "lime"),
    ("key", _key("(h > 0.39) & (h < 0.45) & (s > 0.5)"), "wrong", "screen-left"),
    ("key", _key("(h > 0.3) & (h < 0.45) & (s > 0.9)"), "wrong", "screen-left"),
    ("key", _key(last="alpha.shape"), "wrong", "shape-last"),
    ("key", _KEY_HEAD + "screen = (h > 0.3) & (h < 0.45) & (s > 0.5)\nalpha = 1 - screen\n", "wrong", "no-value"),
    ("key", _KEY_START.replace("alpha = ...", "alpha = 1 - screen").replace(
        "screen = ...", "screen = (h > 0.3) & (h < 0.45) & (s > 0.5)\nx = ..."), "wrong", "dots-left"),
]
