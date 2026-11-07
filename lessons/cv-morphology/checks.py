# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-morphology: which pixels an erosion keeps, dilation with a square of any size, the smallest
closing that fills the holes the shine of the lamp leaves in the apples, and why the opening comes before the
closing. And the pictures: every step around the apple at the top left of the drawing of fruit; and for the slider,
the closing of one size next to how many pixels are wrong at every size."""
import functools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value

FRUIT = "data/photos/fruit.png"          # the drawing of cv-color-spaces: 360 by 240 pixels, RGB
LABELS = "data/photos/fruit-labels.png"  # what every pixel is: 0 the tablecloth, 1 an apple, 2 to 4 other fruit
APPLE = 1
RED = 0.45                               # the saturation threshold of the rule for red in this lesson
OPEN = 3                                 # the square of the opening, in the worked example and on the slider
SIZES = [1, 3, 5, 7, 9, 11, 13, 15]      # the stops of the slider: the square of the closing
CROP = (slice(20, 120), slice(20, 120))  # the part of the drawing around the apple at the top left


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


@functools.lru_cache(maxsize=1)
def red_mask() -> np.ndarray:
    """The rule for red of cv-color-spaces, with 0.45 instead of 0.5: hue near 0 or near 1, and saturation above
    0.45. As the worked example makes it."""
    from matplotlib.colors import rgb_to_hsv
    hsv = rgb_to_hsv(read(FRUIT) / 255)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    mask = ((h < 0.05) | (h > 0.95)) & (s > RED)
    mask.flags.writeable = False
    return mask


@functools.lru_cache(maxsize=1)
def apples() -> np.ndarray:
    """Where the course drew an apple."""
    out = read(LABELS) == APPLE
    out.flags.writeable = False
    return out


def _squares(img, k, outside):
    """Every window of k by k, centered on a pixel, as k * k shifted copies of the image; outside the image, the
    value `outside`, which never changes the outcome: True for an erosion, False for a dilation."""
    img = np.asarray(img, dtype=bool)
    h = k // 2
    p = np.full((img.shape[0] + 2 * h, img.shape[1] + 2 * h), outside)
    p[h:h + img.shape[0], h:h + img.shape[1]] = img
    return [p[i:i + img.shape[0], j:j + img.shape[1]] for i in range(k) for j in range(k)]


def erode_of(img, k: int) -> np.ndarray:
    """Erosion with a square of k by k: white only where the whole window is white. Only the part of the window
    inside the image counts, as with `img[max(r - h, 0):r + h + 1, ...]` in the lesson."""
    return np.logical_and.reduce(_squares(img, k, True))


def dilate_of(img, k: int) -> np.ndarray:
    """Dilation with a square of k by k: white where any pixel of the window is white."""
    return np.logical_or.reduce(_squares(img, k, False))


def opening_of(img, k: int) -> np.ndarray:
    return dilate_of(erode_of(img, k), k)


def closing_of(img, k: int) -> np.ndarray:
    return erode_of(dilate_of(img, k), k)


@functools.lru_cache(maxsize=1)
def opened() -> np.ndarray:
    """The red pixels after the opening of 3 by 3 of the worked example."""
    out = opening_of(red_mask(), OPEN)
    out.flags.writeable = False
    return out


@functools.lru_cache(maxsize=None)
def closed_at(k: int) -> np.ndarray:
    """The opening of the worked example, closed with a square of k by k: what the slider shows."""
    out = closing_of(opened(), int(k))
    out.flags.writeable = False
    return out


def wrong(mask) -> int:
    """How many pixels a mask has wrong: white where the course drew no apple, or black where it drew one."""
    return int((np.asarray(mask, dtype=bool) != apples()).sum())


def missed(mask) -> int:
    """How many apple pixels are black: in this lesson, the holes the shine of the lamp leaves."""
    return int((apples() & ~np.asarray(mask, dtype=bool)).sum())


def shapes_of(mask) -> list[list[tuple[int, int]]]:
    """Every white shape: the pixels that touch, also diagonally. Each as a list of (row, column)."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        todo = [(int(r0), int(c0))]
        for r, c in todo:
            for rr in range(max(r - 1, 0), min(r + 2, mask.shape[0])):
                for cc in range(max(c - 1, 0), min(c + 2, mask.shape[1])):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        todo.append((rr, cc))
        out.append(todo)
    return out


@functools.lru_cache(maxsize=None)
def size_outcome(k: int) -> dict:
    """What the slider leads to at a closing of k by k: the black pixels of the apples, the white shapes and the
    pixels wrong."""
    mask = closed_at(int(k))
    return {"holes": missed(mask), "shapes": len(shapes_of(mask)), "wrong": wrong(mask)}


def _photo_axes(ax, img, title):
    if np.ndim(img) == 3:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(np.asarray(img, dtype=float), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def steps_picture(labels: dict[str, str]) -> str:
    """Six pictures of the part around the apple at the top left: the drawing; the rule for red with 0.45; the
    erosion of 3 by 3; the opening; the dilation of the opening; and the closing of the opening. White is red."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    mask = red_mask()
    eroded = erode_of(mask, OPEN)
    dilated = dilate_of(opened(), 3)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1), dpi=110, gridspec_kw={"hspace": 0.16, "wspace": 0.05})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    for ax, (img, key) in zip(axes.ravel(), [(read(FRUIT), "drawing"), (mask, "red"), (eroded, "erosion"),
                                             (opened(), "opening"), (dilated, "dilation"), (closed_at(3), "closing")]):
        _photo_axes(ax, img[CROP], labels[key])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def size_frame(k: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the opening of the worked example closed with a square of k by k, and next to it how many
    pixels are wrong at every size, with k marked. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    k = int(k)
    curve = [size_outcome(s)["wrong"] for s in SIZES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.22})
    _photo_axes(left, closed_at(k), labels["closed"].format(k=k))
    right.plot(SIZES, curve, color=plots.LOSS, linewidth=1.8, marker="o", markersize=3.5)
    here = curve[SIZES.index(k)]
    right.plot([k, k], [0, here], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([k], [here], marker="o", color=plots.HYPER, markersize=8)
    right.set_xlim(0, 16)
    right.set_ylim(0, 450)
    right.set_xticks(SIZES)
    right.set_xlabel(labels["size"])
    right.set_title(labels["wrong"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(k=k))
    plt.close(fig)
    return html, size_outcome(k)


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _listed(v) -> str:
    """An image the way the lesson writes it: an array as a list of rows of 0 and 1, on one line."""
    a = _numbers(v)
    if a is None or a.ndim != 2:
        return str(v.tolist() if isinstance(v, np.ndarray) else v)
    return str([[int(x) if float(x).is_integer() else round(float(x), 2) for x in row] for row in a])


def _same_image(got, want) -> bool:
    """Right when every pixel is white or black as in the course's answer, in the same shape: True and 1 count as
    the same."""
    a = _numbers(got)
    return a is not None and a.shape == np.shape(want) and bool(np.array_equal(a, np.asarray(want, dtype=float)))


def _never_filled(got, text) -> bool:
    """Only zeros back, and no line that stores something in `out`."""
    a = _numbers(got)
    return (a is not None and a.size > 0 and bool(np.all(a == 0))
            and re.search(r"^[ \t]*out\[", text, re.MULTILINE) is None)


# The lesson is done when these are correct. "block" is a choice question and "order" an open question in the lesson.
REQUIRED = ["block", "dilate", "size", "order"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_HEAD = ("import numpy as np\n\n\n"
         "def dilate(img, k):\n"
         "    img = np.array(img)\n"
         "    half = k // 2\n"
         "    out = np.zeros(img.shape, dtype=bool)\n"
         "    for r in range(len(img)):\n        for c in range(len(img[0])):\n")
_TAIL = "    return out\n"
_START = _HEAD + "            ...\n" + _TAIL


def _with(window: str, store: str = "window.max()") -> str:
    """The course's code with another window, and another value stored."""
    return _HEAD + f"            window = {window}\n            out[r, c] = {store}\n" + _TAIL


_WINDOW = "img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1]"
_SOLUTION = _with(_WINDOW)


def _image(*rows: str) -> list[list[int]]:
    """A small binary image, written as rows of 0 and 1."""
    return [[int(x) for x in row] for row in rows]


# The example of exercise 2: one white pixel in the middle.
DOT = _image("00000", "00000", "00100", "00000", "00000")
_CORNERS = _image("10000", "00000", "00000", "00001")
_DOT7 = _image("0000000", "0000000", "0000000", "0001000", "0000000", "0000000", "0000000")
_RING = _image("000000", "011110", "010010", "010010", "011110", "000000")
_TWO = _image("0100000", "0000010", "0000000")
_PI = (np.array([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3], [5, 8, 9, 7, 9], [3, 2, 3, 8, 4], [6, 2, 6, 4, 3]]) > 5).astype(int).tolist()
_BOOLS = [[False, True, False, False, False, False], [False, False, False, False, False, False],
          [False, False, False, False, False, False], [False, False, False, False, True, False],
          [False, False, False, False, False, False]]

EXERCISES = {
    "dilate": FunctionBox(
        name="dilate",
        function="dilate",
        start=_START,
        solution=_SOLUTION,
        visible=[(DOT, 3)],
        hidden=[(_CORNERS, 3), (_DOT7, 5), (_RING, 3), (_TWO, 3), (_PI, 1), (_PI, 3), (_BOOLS, 5)],
        show=_listed,
        compare=_same_image,
        first=[_DOTS],
        mistakes=[
            Mistake("chained", syntax=r"img\[[^\]\n]*\]\s*\[",
                    en="Your function stops with an error, in this line: `{line}`. `img[...][...]` takes rows twice: the second pair of square brackets picks from the rows you already took. Put the rows and the columns in one pair: `img[rows, columns]`.",
                    nl="Je functie stopt met een fout, in deze regel: `{line}`. `img[...][...]` pakt twee keer rijen: het tweede paar vierkante haken kiest uit de rijen die je al had. Zet de rijen en de kolommen in één paar: `img[rijen, kolommen]`."),
            Mistake("negative", error=lambda e: isinstance(e, ValueError) and "zero-size" in str(e),
                    en="With `{call}`, your function stops with an error: a window is empty. At `r = 0`, `r - half` is below 0, and in Python a slice that starts below 0 counts from the end, so it holds no rows. Start the slice at `max(r - half, 0)`, and the columns at `max(c - half, 0)`.",
                    nl="Met `{call}` stopt je functie met een fout: een venster is leeg. Bij `r = 0` is `r - half` kleiner dan 0, en in Python telt een slice die onder 0 begint vanaf het eind, dus zit er geen rij in. Begin de slice bij `max(r - half, 0)`, en de kolommen bij `max(c - half, 0)`."),
            Mistake("half", error=lambda e: isinstance(e, TypeError) and "slice indices" in str(e),
                    en="With `{call}`, your function stops with an error: a slice needs whole numbers, and `k / 2` gives a float, such as 1.5. `k // 2` divides and rounds down: `3 // 2` is 1, and `5 // 2` is 2.",
                    nl="Met `{call}` stopt je functie met een fout: een slice heeft hele getallen nodig, en `k / 2` geeft een float, zoals 1.5. `k // 2` deelt en rondt naar beneden af: `3 // 2` is 1, en `5 // 2` is 2."),
            Mistake("builtin-max", error=lambda e: isinstance(e, ValueError) and "ambiguous" in str(e),
                    en="With `{call}`, your function stops with an error. Python's own `max(window)` goes through the window row by row, and cannot compare two rows. Use `window.max()`, which looks at every pixel.",
                    nl="Met `{call}` stopt je functie met een fout. Pythons eigen `max(window)` loopt het venster rij voor rij af, en kan twee rijen niet vergelijken. Gebruik `window.max()`, die naar elke pixel kijkt."),
            Mistake("many", error=lambda e: isinstance(e, ValueError) and "sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `out[r, c]` is one place, and it gets a whole array. Store one value in it: the largest value of the window, `window.max()`.",
                    nl="Met `{call}` stopt je functie met een fout: `out[r, c]` is één plek, en die krijgt een hele array. Sla er één waarde in op: de grootste waarde van het venster, `window.max()`."),
            Mistake("dilate-name", error=lambda e: isinstance(e, TypeError) and "item assignment" in str(e),
                    en="With `{call}`, your function stops with an error: inside the function, `dilate` is the function itself, not an array. Here the result is called `out`: store it in `out[r, c]`.",
                    nl="Met `{call}` stopt je functie met een fout: binnen de functie is `dilate` de functie zelf, geen array. De uitkomst heet hier `out`: sla hem op in `out[r, c]`."),
            _m("in-place", "You store the result in `img`, the image you are reading, so the pixels after it see neighbors that have already grown. Store it in `out[r, c]`, and give back `out`.",
               "Je slaat de uitkomst op in `img`, het beeld dat je leest, dus de pixels erna zien buren die al gegroeid zijn. Sla hem op in `out[r, c]`, en geef `out` terug.",
               matches=lambda got, text: re.search(r"^[ \t]*img\[\s*r\s*,\s*c\s*\]\s*=", text, re.MULTILINE) is not None),
            _m("chained", "`img[...][...]` takes rows twice: the second pair of square brackets picks from the rows you already took. Put the rows and the columns in one pair: `img[rows, columns]`.",
               "`img[...][...]` pakt twee keer rijen: het tweede paar vierkante haken kiest uit de rijen die je al had. Zet de rijen en de kolommen in één paar: `img[rijen, kolommen]`.",
               matches=lambda got, text: re.search(r"img\[[^\]\n]*\]\s*\[", text) is not None),
            Mistake("zeros", matches=_never_filled,
                    en="`{call}` gives only zeros: `out` is still the array of zeros it started as. For every pixel, store the largest value of its window in `out[r, c]`.",
                    nl="`{call}` geeft alleen nullen: `out` is nog steeds de array met nullen van het begin. Sla voor elke pixel de grootste waarde van zijn venster op in `out[r, c]`."),
            _m("erode", "You take the smallest value of the window, `min()`: that is an erosion, and a lone white pixel disappears. A dilation takes the largest, `window.max()`: white as soon as one pixel of the window is white.",
               "Je neemt de kleinste waarde van het venster, `min()`: dat is een erosie, en een losse witte pixel verdwijnt. Een dilatatie neemt de grootste, `window.max()`: wit zodra één pixel van het venster wit is.",
               wrong=_with(_WINDOW, "window.min()")),
            _m("swapped", "The first part between the square brackets is the rows, and those go with `r`; the second is the columns, with `c`.",
               "Het eerste deel tussen de vierkante haken is de rijen, en die horen bij `r`; het tweede de kolommen, bij `c`.",
               matches=lambda got, text: re.search(r"img\[\s*max\(\s*c\b", text) is not None),
            _m("copy", "Every pixel keeps its own value, so nothing grows. Store the largest value of the whole window around the pixel: `window.max()`.",
               "Elke pixel houdt zijn eigen waarde, dus er groeit niets. Sla de grootste waarde van het hele venster rond de pixel op: `window.max()`.",
               wrong=_HEAD + "            out[r, c] = img[r, c]\n" + _TAIL),
            _m("three", "Your window is always 3 by 3, whatever `k` is, as in the worked example. The square reaches `half` pixels to each side: the rows run from `max(r - half, 0)` to `r + half + 1`.",
               "Je venster is altijd 3 bij 3, wat `k` ook is, zoals in het uitgewerkte voorbeeld. Het vierkant reikt `half` pixels naar elke kant: de rijen lopen van `max(r - half, 0)` tot `r + half + 1`.",
               wrong=_with("img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]")),
            _m("end", "The white grows to one side only. A slice stops just before its end, so `r + half` leaves out the last row of the window: go to `r + half + 1`, and to `c + half + 1` for the columns.",
               "Het wit groeit maar naar één kant. Een slice stopt net voor zijn eind, dus `r + half` laat de laatste rij van het venster weg: ga tot `r + half + 1`, en tot `c + half + 1` voor de kolommen.",
               wrong=_with("img[max(r - half, 0):r + half, max(c - half, 0):c + half]")),
            _m("shifted", "The white grows up and to the left only: your window starts at the pixel itself and goes down and to the right. The pixel is the middle of its window: from `r - half` to `r + half`.",
               "Het wit groeit alleen naar boven en naar links: je venster begint bij de pixel zelf en gaat naar beneden en naar rechts. De pixel is het midden van zijn venster: van `r - half` tot `r + half`.",
               wrong=_with("img[r:r + k, c:c + k]")),
            _m("whole-k", "The white grows too far: your window goes `k` pixels to each side, so it is `2 * k + 1` wide. A square of `k` by `k` reaches `half = k // 2` pixels to each side.",
               "Het wit groeit te ver: je venster gaat `k` pixels naar elke kant, dus het is `2 * k + 1` breed. Een vierkant van `k` bij `k` reikt `half = k // 2` pixels naar elke kant.",
               wrong=_with("img[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]")),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the window of `k` by `k` around `(r, c)`, and store its largest value in `out[r, c]`.",
                      "nl": "Vervang de `...` door het venster van `k` bij `k` rond `(r, c)`, en sla de grootste waarde ervan op in `out[r, c]`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with an image and the size of the square, and the `def` line of your function has a different number of names between its parentheses. Keep `(img, k)`.",
                          "nl": "De check roept `{call}` aan, met een beeld en de grootte van het vierkant, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. A pixel becomes white when any pixel in the square of `k` by `k` around it is white; only the part of the square inside the image counts.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een pixel wordt wit als er een pixel in het vierkant van `k` bij `k` eromheen wit is; alleen het deel van het vierkant binnen het beeld telt."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your window work for every size `k`, and for a pixel at the border?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je venster voor elke grootte `k`, en voor een pixel aan de rand?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example, the one white pixel grows into a square of 3 by 3: every pixel whose window holds it becomes white. With `k = 5` it would grow into a square of 5 by 5.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld groeit de ene witte pixel uit tot een vierkant van 3 bij 3: elke pixel met die pixel in zijn venster wordt wit. Met `k = 5` zou hij uitgroeien tot een vierkant van 5 bij 5."},
        },
    ),
    "size": Value(
        name="size",
        start=1,
        solution=5,            # the smallest stop with no black apple pixel; the test of the lesson measures it again
        outcome=int,
        prose_numbers=True,
        mistakes=[
            Mistake("small", matches=lambda got, k: int(k) < 5,
                    en="With a square of {value} by {value}, 89 pixels of the holes are still black: a square of {value} by {value} fits inside every hole, so the closing leaves most of it open. Move the slider to the right.",
                    nl="Met een vierkant van {value} bij {value} zijn er nog 89 pixels van de gaten zwart: een vierkant van {value} bij {value} past in elk gat, dus de sluiting laat het grootste deel open. Schuif de slider naar rechts."),
            Mistake("joined", matches=lambda got, k: int(k) >= 11,
                    en="A square of {value} by {value} fills every hole, but the two apples on the right grow together into one shape, and a smaller square fills every hole too. Look for the smallest.",
                    nl="Een vierkant van {value} bij {value} vult elk gat, maar de twee appels rechts groeien aan elkaar vast tot één vorm, en een kleiner vierkant vult ook elk gat. Zoek het kleinste."),
            Mistake("big", matches=lambda got, k: int(k) > 5,
                    en="A square of {value} by {value} fills every hole, but a smaller square does too. Look for the smallest: the bigger the square, the more the closing changes elsewhere.",
                    nl="Een vierkant van {value} bij {value} vult elk gat, maar een kleiner vierkant doet dat ook. Zoek het kleinste: hoe groter het vierkant, hoe meer de sluiting ergens anders verandert."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The square is still 1 by 1: a window of one pixel changes nothing, and 101 pixels of the holes are black. Move the slider to the right.",
                      "nl": "Het vierkant is nog 1 bij 1: een venster van één pixel verandert niets, en 101 pixels van de gaten zijn zwart. Schuif de slider naar rechts."},
            "wrong": {"en": "A square of {value} by {value} is not it. Look for the smallest square after which no pixel of the holes is black.",
                      "nl": "Een vierkant van {value} bij {value} is het niet. Zoek het kleinste vierkant waarna geen pixel van de gaten meer zwart is."},
            "correct": {"en": "{value} by {value} is right: the smallest square that fills every hole. Then 14 pixels are wrong, all of them pixels of the tablecloth that the opening left on an apple. From 9 on, the closing also fills the narrow strip between the small apple and the right edge of the drawing, because the erosion does not look past the edge; and from 11 on the gap between the two apples on the right: they become one shape.",
                        "nl": "{value} bij {value} klopt: het kleinste vierkant dat elk gat vult. Dan zijn er 14 pixels fout, allemaal pixels van het tafelkleed die de opening op een appel liet staan. Vanaf 9 vult de sluiting ook de smalle strook tussen de kleine appel en de rechterrand van de tekening, omdat de erosie niet voorbij de rand kijkt; en vanaf 11 het gat tussen de twee appels rechts: die worden één vorm."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("dilate", _START, "empty", None),
    ("dilate", _SOLUTION, "correct", None),
    ("dilate", _HEAD + "            window = img[max(r - k // 2, 0):r + k // 2 + 1, max(c - k // 2, 0):c + k // 2 + 1]\n"
     "            out[r, c] = window.max()\n" + _TAIL, "correct", None),
    ("dilate", _with(_WINDOW, "window.sum() > 0"), "correct", None),
    ("dilate", _with(_WINDOW, "window.any()"), "correct", None),
    ("dilate", _with(_WINDOW.replace("half, 0)", "int(k / 2), 0)").replace("half + 1", "int(k / 2) + 1")), "correct", None),
    ("dilate", _with("img[max(c - half, 0):c + half + 1, max(r - half, 0):r + half + 1]"), "wrong", "swapped"),
    ("dilate", _HEAD + "            out[r, c] = img[max(c - half, 0):c + half + 1, max(r - half, 0):r + half + 1].max()\n" + _TAIL,
     "wrong", "swapped"),
    ("dilate", _with("img[r - half:r + half + 1, c - half:c + half + 1]"), "error", "negative"),
    ("dilate", _with(_WINDOW).replace("half = k // 2", "half = k / 2"), "error", "half"),
    ("dilate", _with(_WINDOW, "window"), "error", "many"),
    ("dilate", _with(_WINDOW, "max(window)"), "error", "builtin-max"),
    ("dilate", _with("img[max(r - half, 0):r + half + 1][max(c - half, 0):c + half + 1]"), "error", "chained"),
    ("dilate", _with("img[max(r - half, 0):r + half + 1][:, max(c - half, 0):c + half + 1]"), "correct", None),
    ("dilate", _HEAD + "            img[r, c] = img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1].max()\n    return img\n",
     "wrong", "in-place"),
    ("dilate", _HEAD + "            img[r, c] = img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1].max()\n" + _TAIL,
     "wrong", "in-place"),
    ("dilate", _HEAD + "            out[r][c] = img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1].max()\n" + _TAIL,
     "correct", None),
    ("dilate", _HEAD + "            dilate[r, c] = img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1].max()\n"
     + _TAIL, "error", "dilate-name"),
    ("dilate", _HEAD + "            window = img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1]\n" + _TAIL,
     "wrong", "zeros"),
    ("dilate", _with(_WINDOW, "window.min()"), "wrong", "erode"),
    ("dilate", _with(_WINDOW, "window.all()"), "wrong", "erode"),
    ("dilate", _HEAD + "            out[r, c] = img[r, c]\n" + _TAIL, "wrong", "copy"),
    ("dilate", _with("img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]"), "wrong", "three"),
    ("dilate", _with("img[max(r - half, 0):r + half, max(c - half, 0):c + half]"), "wrong", "end"),
    ("dilate", _with("img[r:r + k, c:c + k]"), "wrong", "shifted"),
    ("dilate", _with("img[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]"), "wrong", "whole-k"),
    ("dilate", _with(_WINDOW, "window.mean() > 0.5"), "wrong", None),
    ("dilate", _HEAD + "            ...\n            out[r, c] = 0\n" + _TAIL, "wrong", "dots-left"),
    ("size", 1, "empty", None),
    ("size", 5, "correct", None),
    ("size", 3, "wrong", "small"),
    ("size", 7, "wrong", "big"),
    ("size", 9, "wrong", "big"),
    ("size", 11, "wrong", "joined"),
    ("size", 15, "wrong", "joined"),
]
