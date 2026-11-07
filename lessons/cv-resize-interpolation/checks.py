# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-resize-interpolation: the bilinear value at a position between four pixels, and the coffee
photo made half as big with Pillow, width before height. And the pictures: one row made four times as long with
nearest-neighbor and with bilinear interpolation, and for the slider, a part of the coffee photo made bigger with both.
The choice between the two for a label image and the open question about shrinking are in the lesson itself."""
import functools
import re

import numpy as np

from everything_ai.check import CodeBox, FunctionBox, Mistake

COFFEE = "data/photos/coffee.png"        # the photo of cv-image-files: 400 rows by 600 columns, RGB
LABELS = "data/photos/fruit-labels.png"  # the label of every pixel of the drawing of fruit: 0 the tablecloth, 1 to 4 fruit
SPOON = (slice(230, 260), slice(320, 365))   # the part of the coffee photo around the spoon: 30 rows by 45 columns
FACTORS = [1, 2, 3, 4, 6, 8]             # the stops of the slider: how many times bigger the part of the photo gets
ROW = [40, 120, 200, 60, 140, 180]       # the row of six pixels of the picture in *The idea*


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


@functools.lru_cache(maxsize=1)
def spoon() -> np.ndarray:
    """The part of the coffee photo around the spoon, as the worked example cuts it."""
    return read(COFFEE)[SPOON]


def middles(h: int, n: int) -> np.ndarray:
    """Where the middles of n new pixels lie in a row of h old ones, counted in old pixels from the edge."""
    return (np.arange(n) + 0.5) * h / n


def nearest(img, n_rows: int, n_cols: int) -> np.ndarray:
    """Nearest-neighbor interpolation: every new pixel takes the old pixel whose square holds its middle."""
    img = np.asarray(img)
    rows = middles(len(img), n_rows).astype(int)
    cols = middles(img.shape[1], n_cols).astype(int)
    return img[rows][:, cols]


def _around(h: int, n: int):
    """For n new pixels in a row of h old ones: the two old pixels around each middle, counted from the old middles
    and kept inside the row, and how far the middle lies from the first of the two."""
    y = np.clip(middles(h, n) - 0.5, 0, h - 1)
    first = y.astype(int)
    return first, np.minimum(first + 1, h - 1), y - first


def bilinear(img, n_rows: int, n_cols: int) -> np.ndarray:
    """Bilinear interpolation, as floats: every new pixel mixes the four old pixels around its middle, first along the
    two rows with u, then between them with t."""
    img = np.asarray(img, dtype=float)
    r0, r1, t = _around(len(img), n_rows)
    c0, c1, u = _around(img.shape[1], n_cols)
    extra = (1,) * (img.ndim - 2)
    t = t.reshape(-1, 1, *extra)
    u = u.reshape(1, -1, *extra)
    top = (1 - u) * img[r0][:, c0] + u * img[r0][:, c1]
    bottom = (1 - u) * img[r1][:, c0] + u * img[r1][:, c1]
    return (1 - t) * top + t * bottom


def rounded(out) -> np.ndarray:
    """Back to whole numbers from 0 to 255, rounded, as the recipe does it: `(out + 0.5).astype(np.uint8)`."""
    return (np.asarray(out) + 0.5).astype(np.uint8)


def pillow(img, n_rows: int, n_cols: int, how: str) -> np.ndarray:
    """What Pillow makes of it: `Image.fromarray(img).resize((n_cols, n_rows), Image.NEAREST or Image.BILINEAR)`."""
    from PIL import Image
    return np.array(Image.fromarray(np.asarray(img)).resize((n_cols, n_rows), getattr(Image, how)))


def row_picture(labels: dict[str, str]) -> str:
    """The row of six pixels made four times as long: the old pixels at their middles, nearest neighbor as steps, and
    linear interpolation, bilinear interpolation in one row, as points on straight lines from middle to middle."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    n = 4 * len(ROW)
    row = np.array([ROW])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    edges = np.arange(n + 1) * len(ROW) / n
    ax.stairs(nearest(row, 1, n)[0], edges, baseline=None, color=plots.INK_3, linewidth=2.2, linestyle="-",
              label=labels["nearest"])
    ax.plot(middles(len(ROW), n), bilinear(row, 1, n)[0], color=plots.INK_2, linestyle="-", linewidth=1.2,
            marker="o", markersize=4, markeredgewidth=0, label=labels["bilinear"])
    ax.plot(np.arange(len(ROW)) + 0.5, ROW, linestyle="none", marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=plots.INK, markeredgewidth=1.6, zorder=3, label=labels["old"])
    ax.set_xlim(0, len(ROW))
    ax.set_ylim(0, 255)
    ax.set_xticks(range(len(ROW) + 1))
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


@functools.lru_cache(maxsize=None)
def zoomed(k: int) -> tuple[np.ndarray, np.ndarray]:
    """The part around the spoon made k times bigger, with nearest neighbor and with bilinear interpolation."""
    k = int(k)
    rows, cols = len(spoon()) * k, spoon().shape[1] * k
    return nearest(spoon(), rows, cols), rounded(bilinear(spoon(), rows, cols))


def zoom_frame(k: int, labels: dict[str, str]) -> str:
    """For the slider: the part around the spoon made k times bigger, nearest neighbor on the left and bilinear
    interpolation on the right, both drawn at the same size."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    k = int(k)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45), dpi=DPI)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.04)
    for ax, img, title in zip(axes, zoomed(k), (labels["nearest"], labels["bilinear"])):
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=10)
    rows, cols = zoomed(k)[0].shape[:2]
    html = plots.render(fig, alt=labels["alt"].format(k=k, rows=rows, cols=cols))
    plt.close(fig)
    return html


# The pictures of the slider are drawn sharper than the others, so that the blocks of nearest neighbor and the soft
# steps of bilinear interpolation stay visible when the part of the photo gets big.
DPI = 160

REQUIRED = ["labels", "value", "half", "shrink"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


# Exercise 2: the bilinear value at row position y and column position x, counted from the middles of the pixels
_HEAD = ("def value_at(img, y, x):\n"
         "    r0, c0 = int(y), int(x)\n"
         "    r1 = min(r0 + 1, len(img) - 1)\n"
         "    c1 = min(c0 + 1, len(img[0]) - 1)\n"
         "    t, u = y - r0, x - c0\n")
_TOP = "(1 - u) * img[r0][c0] + u * img[r0][c1]"
_BOTTOM = "(1 - u) * img[r1][c0] + u * img[r1][c1]"
_BACK = "(1 - t) * top + t * bottom"
_START = _HEAD + "    top = ...\n    bottom = ...\n    return ...\n"


def _with(top: str = _TOP, bottom: str = _BOTTOM, back: str = _BACK) -> str:
    """The course's function with another mix along the rows, or another mix between them."""
    return _HEAD + f"    top = {top}\n    bottom = {bottom}\n    return {back}\n"


_SOLUTION = _with()

# The example of exercise 2, and the example in *The idea*
TWO = [[0, 100],
       [200, 40]]
IDEA = [[0, 80],
        [40, 200]]
_THREE = [[10, 50, 90],
          [30, 70, 250],
          [200, 0, 100]]
_ONE_ROW = [[0, 100, 200]]
_ONE_COLUMN = [[10], [30], [50]]
_PHOTO = np.array([[200, 40], [10, 250]], dtype=np.uint8)
_COLOR = np.array([[[250, 10, 0], [10, 250, 0]], [[0, 0, 250], [250, 250, 250]]], dtype=np.uint8)

# Exercise 3: the coffee photo half as big, with Pillow
_HALF_HEAD = ("import numpy as np\n"
              "from PIL import Image\n\n"
              "coffee = np.array(Image.open(\"data/photos/coffee.png\"))\n")
_HALF_START = _HALF_HEAD + "small = ...\nsmall\n"


def _half(line: str) -> str:
    """The box of exercise 3 with this line for `small`."""
    return _HALF_HEAD + f"small = {line}\nsmall\n"


_HALF_SOLUTION = _half("np.array(Image.fromarray(coffee).resize((300, 200), Image.BILINEAR))")


def _shape_is(*shape):
    return lambda got, text: isinstance(got, np.ndarray) and got.shape == shape


def _is_image(got, text) -> bool:
    from PIL import Image
    return isinstance(got, Image.Image)


EXERCISES = {
    "value": FunctionBox(
        name="value",
        function="value_at",
        start=_START,
        solution=_SOLUTION,
        visible=[(TWO, 0.25, 0.5)],
        hidden=[(TWO, 0.5, 0.25), (TWO, 0.0, 0.0), (TWO, 1.0, 1.0), (TWO, 1.0, 0.5), (_THREE, 1.25, 0.75),
                (_THREE, 0.5, 1.5), (_ONE_ROW, 0.0, 1.5), (_ONE_COLUMN, 1.5, 0.0), (_PHOTO, 0.5, 0.5),
                (_PHOTO, 0.25, 0.75), (_COLOR, 0.5, 0.25)],
        first=[_DOTS],
        mistakes=[
            Mistake("list-index", error=lambda e: isinstance(e, TypeError) and "list indices" in str(e),
                    en="With `{call}`, your function stops with an error: `img` is a list of rows here, and a list takes one number between square brackets. Write `img[r0][c0]`: first the row, then the column in it. That works for an array too.",
                    nl="Met `{call}` stopt je functie met een fout: `img` is hier een lijst van rijen, en een lijst neemt één getal tussen de vierkante haken. Schrijf `img[r0][c0]`: eerst de rij, dan de kolom daarin. Dat werkt ook voor een array."),
            Mistake("border", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an error: it asks for a row or a column past the last one. At the last row there is no row below it. `r1` and `c1` stay inside the image: use them for the second row and the second column.",
                    nl="Met `{call}` stopt je functie met een fout: hij vraagt om een rij of kolom na de laatste. Bij de laatste rij is er geen rij eronder. `r1` en `c1` blijven binnen het beeld: gebruik die voor de tweede rij en de tweede kolom."),
            _m("flip", "The weights are the wrong way around: the farther pixel counts most. The nearer a position lies to a pixel, the more that pixel counts: `(1 - u)` for the left pixel, `c0`, and `u` for the right one, `c1`; `(1 - t)` for `top` and `t` for `bottom`.",
               "De gewichten staan omgekeerd: de verste pixel telt het zwaarst. Hoe dichter een positie bij een pixel ligt, hoe zwaarder die pixel telt: `(1 - u)` voor de linkerpixel, `c0`, en `u` voor de rechter, `c1`; `(1 - t)` voor `top` en `t` voor `bottom`.",
               wrong=_with("u * img[r0][c0] + (1 - u) * img[r0][c1]", "u * img[r1][c0] + (1 - u) * img[r1][c1]",
                           "t * top + (1 - t) * bottom")),
            _m("flip-u", "Along the rows, the weights are the wrong way around: the farther pixel counts most. `u` is how far the position lies to the right of `c0`, so the left pixel, `img[r0][c0]`, gets `(1 - u)`, and the right one `u`.",
               "Langs de rijen staan de gewichten omgekeerd: de verste pixel telt het zwaarst. `u` is hoe ver de positie rechts van `c0` ligt, dus de linkerpixel, `img[r0][c0]`, krijgt `(1 - u)`, en de rechter `u`.",
               wrong=_with("u * img[r0][c0] + (1 - u) * img[r0][c1]", "u * img[r1][c0] + (1 - u) * img[r1][c1]")),
            _m("flip-t", "Between the rows, the weights are the wrong way around: the farther row counts most. `t` is how far the position lies below `r0`, so `top` gets `(1 - t)`, and `bottom` gets `t`.",
               "Tussen de rijen staan de gewichten omgekeerd: de verste rij telt het zwaarst. `t` is hoe ver de positie onder `r0` ligt, dus `top` krijgt `(1 - t)`, en `bottom` krijgt `t`.",
               wrong=_with(back="t * top + (1 - t) * bottom")),
            _m("swap", "`t` and `u` have changed places. `u` is how far the position lies to the right of `c0`: it mixes the two pixels of one row. `t` is how far it lies below `r0`: it mixes `top` and `bottom`.",
               "`t` en `u` zijn van plaats gewisseld. `u` is hoe ver de positie rechts van `c0` ligt: die mengt de twee pixels van één rij. `t` is hoe ver hij onder `r0` ligt: die mengt `top` en `bottom`.",
               wrong=_with("(1 - t) * img[r0][c0] + t * img[r0][c1]", "(1 - t) * img[r1][c0] + t * img[r1][c1]",
                           "(1 - u) * top + u * bottom")),
            _m("mean", "That is the mean of the four pixels around the position, wherever it lies. Bilinear interpolation weighs them: the nearer pixel counts more, with `(1 - u)` and `u` along the rows, and `(1 - t)` and `t` between them.",
               "Dat is het gemiddelde van de vier pixels rond de positie, waar die ook ligt. Bilineaire interpolatie weegt ze: de dichterbij gelegen pixel telt zwaarder, met `(1 - u)` en `u` langs de rijen, en `(1 - t)` en `t` ertussen.",
               wrong=_with("(img[r0][c0] + img[r0][c1]) / 2", "(img[r1][c0] + img[r1][c1]) / 2", "(top + bottom) / 2")),
            _m("top-only", "Your value stays on the top row: the row `r1` below it does not count. Give back the mix of `top` and `bottom`, with `t`: `(1 - t) * top + t * bottom`.",
               "Je waarde blijft op de bovenste rij: de rij `r1` eronder telt niet mee. Geef de mix van `top` en `bottom` terug, met `t`: `(1 - t) * top + t * bottom`.",
               wrong=_with(back="top")),
            _m("position", "Your weights come from the position itself, `y` and `x`, and not from how far it lies from `r0` and `c0`. From row 1 on that goes wrong: at `y = 1.25` the weights would be -0.25 and 1.25. Use `t` and `u`, which always lie between 0 and 1.",
               "Je gewichten komen van de positie zelf, `y` en `x`, en niet van hoe ver die van `r0` en `c0` ligt. Vanaf rij 1 gaat dat mis: bij `y = 1.25` zouden de gewichten -0,25 en 1,25 zijn. Gebruik `t` en `u`, die altijd tussen 0 en 1 liggen.",
               wrong=_with("(1 - x) * img[r0][c0] + x * img[r0][c1]", "(1 - x) * img[r1][c0] + x * img[r1][c1]",
                           "(1 - y) * top + y * bottom")),
            _m("wrap", "For a list your function is right, but this is a photo of uint8, and `img[r0][c1] - img[r0][c0]` cannot go below 0: 40 - 200 wraps around to 96, as in [From light to a file of numbers](/course/open/cv-image-files?lang=en). Use the form with `(1 - u)` and `u`, which only adds, or turn the pixels into ordinary numbers first.",
               "Voor een lijst klopt je functie, maar dit is een foto van uint8, en `img[r0][c1] - img[r0][c0]` kan niet onder 0: 40 - 200 loopt rond naar 96, zoals in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl). Gebruik de vorm met `(1 - u)` en `u`, die alleen optelt, of maak er eerst gewone getallen van.",
               wrong=_with("img[r0][c0] + u * (img[r0][c1] - img[r0][c0])", "img[r1][c0] + u * (img[r1][c1] - img[r1][c0])",
                           "top + t * (bottom - top)")),
            _m("nearest", "That is the value of one pixel: nearest-neighbor interpolation. Bilinear interpolation mixes the four pixels around the position: along the two rows with `u`, then between them with `t`.",
               "Dat is de waarde van één pixel: nearest-neighbor-interpolatie. Bilineaire interpolatie mengt de vier pixels rond de positie: langs de twee rijen met `u`, en daarna ertussen met `t`.",
               wrong=_HEAD + "    return img[int(y + 0.5)][int(x + 0.5)]\n"),
            _m("rounded", "Your function gives a whole number. Bilinear interpolation gives a value between the pixels, with what comes after the point too, so do not round inside the function: the recipe rounds once, at the end, for the whole image.",
               "Je functie geeft een geheel getal. Bilineaire interpolatie geeft een waarde tussen de pixels, met wat na de komma komt erbij, dus rond niet af in de functie: het recept rondt één keer af, aan het eind, voor het hele beeld.",
               matches=lambda got, text: (isinstance(got, (int, np.integer)) and not isinstance(got, bool))
               or re.search(r"\bround\(", text) is not None),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: `top` mixes the two pixels of row `r0`, `img[r0][c0]` and `img[r0][c1]`, with `u`; `bottom` does the same for row `r1`; and the function gives back the mix of `top` and `bottom`, with `t`.",
                      "nl": "Vervang elke `...`: `top` mengt de twee pixels van rij `r0`, `img[r0][c0]` en `img[r0][c1]`, met `u`; `bottom` doet hetzelfde voor rij `r1`; en de functie geeft de mix van `top` en `bottom` terug, met `t`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with an image and a position, and the `def` line of your function has a different number of names between its parentheses. Keep `(img, y, x)`.",
                          "nl": "De check roept `{call}` aan, met een beeld en een positie, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, y, x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Mix along each row with `u`, `(1 - u) * left + u * right`, and then between `top` and `bottom` with `t`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Meng langs elke rij met `u`, `(1 - u) * links + u * rechts`, en daarna tussen `top` en `bottom` met `t`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work at the last row and column, for a photo of uint8, and for a color image, where a pixel is three numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie bij de laatste rij en kolom, voor een foto van uint8, en voor een kleurenbeeld, waarin een pixel drie getallen is?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example, halfway along the top row lies 50, and halfway along the bottom row 120; a quarter of the way from 50 to 120 is 67.5. The same lines work for a color image: there a pixel is an array of three numbers, and NumPy mixes all three at once.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld ligt halverwege de bovenste rij 50, en halverwege de onderste rij 120; op een kwart van 50 naar 120 ligt 67,5. Dezelfde regels werken voor een kleurenbeeld: daar is een pixel een array van drie getallen, en NumPy mengt ze alle drie tegelijk."},
        },
    ),
    "half": CodeBox(
        name="half",
        start=_HALF_START,
        solution=_HALF_SOLUTION,
        first=[_DOTS],
        mistakes=[
            Mistake("float-size", error=lambda e: isinstance(e, TypeError) and "'float' object cannot be interpreted" in str(e),
                    en="Pillow wants whole numbers for the size, and `600 / 2` is `300.0`, a float. Write the size as whole numbers: 300 and 200.",
                    nl="Pillow wil gehele getallen voor de grootte, en `600 / 2` is `300.0`, een float. Schrijf de grootte als gehele getallen: 300 en 200."),
            Mistake("array-resize", error=lambda e: isinstance(e, TypeError) and "'tuple' object cannot be interpreted" in str(e),
                    en="A NumPy array has a `resize` of its own, which only changes the shape of its list of numbers, without interpolation. Make a Pillow image of it first: `Image.fromarray(coffee).resize(...)`.",
                    nl="Een NumPy-array heeft een eigen `resize`, die alleen de vorm van zijn rij getallen verandert, zonder interpolatie. Maak er eerst een Pillow-afbeelding van: `Image.fromarray(coffee).resize(...)`."),
            Mistake("size-shape", error=lambda e: isinstance(e, TypeError) and "sequence of length 2" in str(e),
                    en="Pillow wants the size as two numbers, (width, height), and a shape has three: rows, columns and colors, and in the other order. Half of 600 columns by 400 rows is `(300, 200)`.",
                    nl="Pillow wil de grootte als twee getallen, (breedte, hoogte), en een shape heeft er drie: rijen, kolommen en kleuren, en in de andere volgorde. De helft van 600 kolommen bij 400 rijen is `(300, 200)`."),
            Mistake("image", matches=_is_image,
                    en="That is a Pillow image, not an array: `resize` gives back a new Pillow image. Make an array of it with `np.array(...)`, as in [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
                    nl="Dat is een Pillow-afbeelding, geen array: `resize` geeft een nieuwe Pillow-afbeelding terug. Maak er een array van met `np.array(...)`, zoals in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl)."),
            Mistake("swapped", matches=_shape_is(300, 200, 3),
                    en="Your image has 300 rows and 200 columns: it stands on its side. Pillow writes a size as (width, height): first the columns, then the rows. Half of 600 columns by 400 rows is `(300, 200)`.",
                    nl="Je afbeelding heeft 300 rijen en 200 kolommen: hij staat op zijn kant. Pillow schrijft een grootte als (breedte, hoogte): eerst de kolommen, dan de rijen. De helft van 600 kolommen bij 400 rijen is `(300, 200)`."),
            Mistake("same-size", matches=_shape_is(400, 600, 3),
                    en="Your image is still 400 by 600 pixels. `resize` does not change the image itself: it gives back a new one. Put that new one into `small`.",
                    nl="Je afbeelding is nog 400 bij 600 pixels. `resize` verandert de afbeelding zelf niet: hij geeft een nieuwe terug. Zet die nieuwe in `small`."),
            Mistake("shape-last", matches=lambda got, text: isinstance(got, tuple) and tuple(got) == (200, 300, 3),
                    en="The size is right. The check wants the image itself, so end the box with the line `small`, as it was.",
                    nl="De grootte klopt. De check wil de afbeelding zelf, dus eindig het vak met de regel `small`, zoals hij was."),
            Mistake("np-resize", matches=lambda got, text: _shape_is(200, 300, 3)(got, text) and "np.resize" in text,
                    en="The shape is right, but `np.resize` does not interpolate: it puts the numbers of the photo in their order into the new shape and drops the rest. Make a Pillow image of `coffee` first, and use the `resize` of that: `Image.fromarray(coffee).resize(...)`.",
                    nl="De shape klopt, maar `np.resize` interpoleert niet: hij zet de getallen van de foto op volgorde in de nieuwe shape en laat de rest weg. Maak eerst een Pillow-afbeelding van `coffee`, en gebruik de `resize` daarvan: `Image.fromarray(coffee).resize(...)`."),
            Mistake("every-second", matches=lambda got, text: _shape_is(200, 300, 3)(got, text) and "::2" in text,
                    en="The size is right, but every second pixel keeps one old pixel of every four and drops the other three, like nearest neighbor. The exercise asks for Pillow's bilinear interpolation: give `resize` the method `Image.BILINEAR`.",
                    nl="De grootte klopt, maar om de pixel houdt één oude pixel van elke vier over en laat de andere drie weg, zoals nearest neighbor. De opdracht vraagt om de bilineaire interpolatie van Pillow: geef `resize` de methode `Image.BILINEAR`."),
            Mistake("nearest-method", matches=lambda got, text: _shape_is(200, 300, 3)(got, text) and "NEAREST" in text,
                    en="The size is right, but `Image.NEAREST` is nearest neighbor: every new pixel takes one old pixel. The exercise asks for bilinear interpolation: `Image.BILINEAR`.",
                    nl="De grootte klopt, maar `Image.NEAREST` is nearest neighbor: elke nieuwe pixel neemt één oude pixel. De opdracht vraagt om bilineaire interpolatie: `Image.BILINEAR`."),
            Mistake("method", matches=lambda got, text: _shape_is(200, 300, 3)(got, text) and "BILINEAR" not in text,
                    en="The size is right, but your code does not name `Image.BILINEAR`. Without a method, Pillow uses its default, bicubic interpolation. Give `resize` the method as its second value: `.resize((300, 200), Image.BILINEAR)`.",
                    nl="De grootte klopt, maar je code noemt `Image.BILINEAR` niet. Zonder methode gebruikt Pillow zijn standaard, bicubische interpolatie. Geef `resize` de methode als tweede waarde: `.resize((300, 200), Image.BILINEAR)`."),
            Mistake("other-numbers", matches=_shape_is(200, 300, 3),
                    en="The size is right, but the numbers are not those of Pillow's bilinear interpolation of the coffee photo. Make a Pillow image of `coffee` itself, and resize it with `Image.BILINEAR`: `Image.fromarray(coffee).resize((300, 200), Image.BILINEAR)`.",
                    nl="De grootte klopt, maar de getallen zijn niet die van de bilineaire interpolatie van Pillow op de koffiefoto. Maak een Pillow-afbeelding van `coffee` zelf, en verklein die met `Image.BILINEAR`: `Image.fromarray(coffee).resize((300, 200), Image.BILINEAR)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the coffee photo, half as big: make a Pillow image of `coffee`, call its `resize` with the new size and `Image.BILINEAR`, and make an array of the result.",
                      "nl": "Vervang de `...` door de koffiefoto, half zo groot: maak een Pillow-afbeelding van `coffee`, roep de methode `resize` daarvan aan met de nieuwe grootte en `Image.BILINEAR`, en maak van de uitkomst een array."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box should be `small`, so the check sees the image. Put it back.",
                         "nl": "De laatste regel van het vak moet `small` zijn, zodat de check de afbeelding ziet. Zet hem terug."},
            "wrong": {"en": "That is not the coffee photo at half its size. `small` should be an array of 200 rows, 300 columns and 3 colors, made with Pillow's `resize` and `Image.BILINEAR`.",
                      "nl": "Dat is niet de koffiefoto op de helft van zijn grootte. `small` moet een array zijn van 200 rijen, 300 kolommen en 3 kleuren, gemaakt met `resize` van Pillow en `Image.BILINEAR`."},
            "correct": {"en": "Right: an array of shape `(200, 300, 3)`, 200 rows by 300 columns. Pillow writes the same size the other way around, `(300, 200)`: first the width, then the height.",
                        "nl": "Goed: een array met shape `(200, 300, 3)`, 200 rijen bij 300 kolommen. Pillow schrijft dezelfde grootte andersom, `(300, 200)`: eerst de breedte, dan de hoogte."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("value", _START, "empty", None),
    ("value", _SOLUTION, "correct", None),
    ("value", _with(back="top + t * (bottom - top)"), "correct", None),
    ("value", _HEAD + "    a = img[r0][c0]\n    b = img[r0][c1]\n    c = img[r1][c0]\n    d = img[r1][c1]\n"
     "    return (1 - t) * (1 - u) * a + (1 - t) * u * b + t * (1 - u) * c + t * u * d\n", "correct", None),
    ("value", _with("img[r0][c0] * (1 - u) + img[r0][c1] * u", "img[r1][c0] * (1 - u) + img[r1][c1] * u",
                    "top * (1 - t) + bottom * t"), "correct", None),
    ("value", _with("(1 - u) * img[r0, c0] + u * img[r0, c1]", "(1 - u) * img[r1, c0] + u * img[r1, c1]"), "error", "list-index"),
    ("value", _with("(1 - u) * img[r0][c0] + u * img[r0][c0 + 1]", "(1 - u) * img[r0 + 1][c0] + u * img[r0 + 1][c0 + 1]"),
     "error", "border"),
    ("value", _with("u * img[r0][c0] + (1 - u) * img[r0][c1]", "u * img[r1][c0] + (1 - u) * img[r1][c1]",
                    "t * top + (1 - t) * bottom"), "wrong", "flip"),
    ("value", _with("u * img[r0][c0] + (1 - u) * img[r0][c1]", "u * img[r1][c0] + (1 - u) * img[r1][c1]"), "wrong", "flip-u"),
    ("value", _with(back="t * top + (1 - t) * bottom"), "wrong", "flip-t"),
    ("value", _with("(1 - t) * img[r0][c0] + t * img[r0][c1]", "(1 - t) * img[r1][c0] + t * img[r1][c1]",
                    "(1 - u) * top + u * bottom"), "wrong", "swap"),
    ("value", _with("(img[r0][c0] + img[r0][c1]) / 2", "(img[r1][c0] + img[r1][c1]) / 2", "(top + bottom) / 2"),
     "wrong", "mean"),
    ("value", _with(back="top"), "wrong", "top-only"),
    ("value", _with("(1 - x) * img[r0][c0] + x * img[r0][c1]", "(1 - x) * img[r1][c0] + x * img[r1][c1]",
                    "(1 - y) * top + y * bottom"), "wrong", "position"),
    ("value", _with("img[r0][c0] + u * (img[r0][c1] - img[r0][c0])", "img[r1][c0] + u * (img[r1][c1] - img[r1][c0])",
                    "top + t * (bottom - top)"), "wrong", "wrap"),
    ("value", _HEAD + "    return img[int(y + 0.5)][int(x + 0.5)]\n", "wrong", "nearest"),
    ("value", _with(back="round((1 - t) * top + t * bottom)"), "wrong", "rounded"),
    ("value", _with(back="int((1 - t) * top + t * bottom)"), "wrong", "rounded"),
    ("value", _with(back="(1 - t) * top + t * bottom\n    print(top)"), "correct", None),
    ("value", _with(back="...").replace("    return ...\n", "    print((1 - t) * top + t * bottom)\n"), "wrong", "prints"),
    ("value", _START.replace("    return ...\n", "    return (1 - t) * top + t * bottom\n"), "wrong", "dots-left"),
    ("half", _HALF_START, "empty", None),
    ("half", _HALF_SOLUTION, "correct", None),
    ("half", _half("np.array(Image.fromarray(coffee).resize((coffee.shape[1] // 2, coffee.shape[0] // 2), Image.BILINEAR))"),
     "correct", None),
    ("half", _half("np.array(Image.fromarray(coffee).resize((300, 200), resample=Image.Resampling.BILINEAR))"), "correct", None),
    ("half", _half("np.array(Image.fromarray(coffee).resize((600 / 2, 400 / 2), Image.BILINEAR))"), "error", "float-size"),
    ("half", _half("coffee.resize((300, 200), Image.BILINEAR)"), "error", "array-resize"),
    ("half", _half("np.array(Image.fromarray(coffee).resize((200, 300, 3), Image.BILINEAR))"), "error", "size-shape"),
    ("half", _half("np.array(Image.fromarray(coffee).resize(coffee.shape, Image.BILINEAR))"), "error", "size-shape"),
    ("half", _half("Image.fromarray(coffee).resize((300, 200), Image.BILINEAR)"), "wrong", "image"),
    ("half", _half("np.array(Image.fromarray(coffee).resize((200, 300), Image.BILINEAR))"), "wrong", "swapped"),
    ("half", _half("np.array(Image.fromarray(coffee))"), "wrong", "same-size"),
    ("half", _HALF_HEAD + "small = np.array(Image.fromarray(coffee).resize((300, 200), Image.BILINEAR))\nsmall.shape\n",
     "wrong", "shape-last"),
    ("half", _half("np.array(Image.fromarray(coffee).resize((300, 200)))"), "wrong", "method"),
    ("half", _half("np.array(Image.fromarray(coffee).resize((300, 200), Image.BICUBIC))"), "wrong", "method"),
    ("half", _half("np.array(Image.fromarray(coffee).resize((300, 200), Image.NEAREST))"), "wrong", "nearest-method"),
    ("half", _half("np.resize(coffee, (200, 300, 3))"), "wrong", "np-resize"),
    ("half", _half("coffee[::2, ::2]"), "wrong", "every-second"),
    ("half", _half("np.array(Image.fromarray(coffee[::-1]).resize((300, 200), Image.BILINEAR))"), "wrong", "other-numbers"),
    ("half", _HALF_HEAD + "small = np.array(Image.fromarray(coffee).resize((300, 200), Image.BILINEAR))\n", "wrong", "no-value"),
    ("half", _HALF_HEAD + "small = ...\nsmall = np.array(Image.fromarray(coffee).resize((300, 200), Image.BILINEAR))\nsmall\n",
     "wrong", "dots-left"),
]
