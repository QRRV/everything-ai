# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-otsu-threshold: which change of an image is a point operation, a binary image made with a
threshold, the threshold of the photo of the cell where the within-group variance is smallest, and Otsu's method
itself. And the pictures: the photo of the cell with its histogram and two binary images, one with the mean gray as
the threshold and one with Otsu's threshold; and for the slider, the binary image of one threshold next to the
within-group variance of every threshold."""
import functools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value
from everything_ai import yours

CELL = "data/photos/cell.png"           # 550 by 660 pixels, grayscale; CC0, from the sample data of scikit-image
THRESHOLDS = list(range(66, 187, 8))    # the stops of the slider: 66, 74, ..., 186


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def cell_photo() -> np.ndarray:
    """The photo of the lesson: a cell in salt water, under a microscope; one gray per pixel."""
    return read(CELL)


def withins_of(img) -> tuple[list[int], np.ndarray]:
    """Every threshold from the darkest gray of the image up to the lightest, the lightest not included, and the
    within-group variance of each, as the worked example works it out: the share of the dark pixels times their
    variance, plus the share of the light pixels times theirs. Worked out from the histogram, which is faster and
    gives the same numbers."""
    img = np.asarray(img)
    counts = np.bincount(img.astype(np.int64).ravel(), minlength=256).astype(float)
    grays = np.arange(len(counts))
    n = counts.sum()
    thresholds = list(range(int(img.min()), int(img.max())))
    withins = []
    for t in thresholds:
        c_dark, c_light = counts[:t + 1], counts[t + 1:]
        g_dark, g_light = grays[:t + 1], grays[t + 1:]
        m_dark = (g_dark * c_dark).sum() / c_dark.sum()
        m_light = (g_light * c_light).sum() / c_light.sum()
        withins.append((((g_dark - m_dark) ** 2 * c_dark).sum() + ((g_light - m_light) ** 2 * c_light).sum()) / n)
    return thresholds, np.array(withins)


def otsu_of(img) -> int:
    """Otsu's threshold: the threshold with the smallest within-group variance; the darkest one if several tie."""
    thresholds, withins = withins_of(img)
    return thresholds[int(np.argmin(withins))]


@functools.lru_cache(maxsize=1)
def _cell_withins() -> tuple[tuple[int, ...], np.ndarray]:
    """The within-group variance of every threshold of the photo of the cell, worked out once: the slider draws a
    picture for every stop."""
    thresholds, withins = withins_of(cell_photo())
    withins.flags.writeable = False
    return tuple(thresholds), withins


def cell_threshold() -> int:
    """Otsu's threshold of the photo of the cell: the answer of exercise 3."""
    thresholds, withins = _cell_withins()
    return thresholds[int(np.argmin(withins))]


def within_at(t) -> float:
    """What the slider leads to: the within-group variance of the photo of the cell at threshold t, to one decimal,
    as the sentence under the picture shows it."""
    thresholds, withins = _cell_withins()
    return round(float(withins[thresholds.index(int(t))]), 1)


def _photo_axes(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def threshold_picture(labels: dict[str, str]) -> str:
    """At the top, the photo of the cell and the binary image of two thresholds: the mean gray and Otsu's threshold.
    Under them, the histogram of the photo, cut off at 300 pixels, with both thresholds marked."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    img = cell_photo()
    mean, best = float(img.mean()), cell_threshold()
    fig = plt.figure(figsize=(7.2, 6.0), dpi=110)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.12)
    _photo_axes(fig.add_subplot(grid[0, 0]), img, labels["photo"])
    _photo_axes(fig.add_subplot(grid[0, 1]), 255 * (img > mean), labels["by_mean"])
    _photo_axes(fig.add_subplot(grid[0, 2]), 255 * (img > best), labels["by_otsu"])
    ax = fig.add_subplot(grid[1, :])
    ax.bar(range(256), np.bincount(img.ravel(), minlength=256), width=1.0, color=plots.DATA)
    ax.axvline(mean, color=plots.INK, linestyle=(0, (4, 3)), linewidth=2.0, zorder=3)
    ax.axvline(best, color=plots.INK, linewidth=1.6)
    for x, text in ((mean, labels["mean_line"]), (best, labels["otsu_line"])):
        ax.text(x, 306, text, ha="center", va="bottom", fontsize=9, color=plots.INK, clip_on=False)
    ax.set_xlim(-3, 258)
    ax.set_ylim(0, 300)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xlabel(labels["gray"])
    ax.set_title(labels["histogram"], fontsize=10, loc="left", pad=20)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def threshold_frame(t: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the binary image of the photo of the cell at threshold t, and next to it the within-group
    variance of every threshold, with t marked. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    img = cell_photo()
    thresholds, withins = _cell_withins()
    within = within_at(t)
    white = float((img > t).mean())
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.0, 1.6], "wspace": 0.25})
    _photo_axes(left, 255 * (img[::2, ::2] > t), labels["binary"].format(t=t))
    right.plot(thresholds, withins, color=plots.LOSS, linewidth=1.8)
    right.plot([t, t], [0, within], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([t], [within], marker="o", color=plots.PARAM, markersize=7)
    right.set_xlim(0, 255)
    right.set_ylim(0, 600)
    right.set_xticks([0, 64, 128, 192, 255])
    right.set_xlabel(labels["threshold"])
    right.set_title(labels["within"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(t=t))
    plt.close(fig)
    return html, {"within": within, "white": white}


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _listed(v) -> str:
    """An image the way the lesson writes it: an array as a list of rows, on one line."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    return str(v)


def _same_image(got, want) -> bool:
    """Exercise 2 counts as right when every pixel has the same gray as the course's, whatever the dtype; True and
    False are not grays."""
    if np.asarray(got).dtype == bool:
        return False
    a = _numbers(got)
    return a is not None and a.shape == np.shape(want) and bool(np.all(a == np.asarray(want, dtype=float)))


def _is_true_false(got) -> bool:
    return np.asarray(got).dtype == bool


def _zeros_and_ones(got) -> bool:
    a = _numbers(got)
    return a is not None and a.size > 0 and bool(np.all((a == 0) | (a == 1))) and bool(np.any(a == 1))


# The lesson is done when these are correct. "point" is a choice question in the lesson.
REQUIRED = ["point", "binary", "lowest", "otsu"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_BIN_HEAD = "import numpy as np\n\n\ndef binary(img, t):\n    img = np.array(img)\n"
_BIN_START = _BIN_HEAD + "    ...\n"
_BIN = _BIN_HEAD + "    return 255 * (img > t)"
# The example of exercise 2: six pixels and threshold 100.
SMALL = [[30, 120, 200], [90, 160, 60]]

_OTSU_HEAD = ("import numpy as np\n\n\ndef otsu(img):\n    img = np.array(img)\n"
              "    thresholds = list(range(img.min(), img.max()))\n    withins = []\n    for t in thresholds:\n")
_OTSU_TAIL = "    return thresholds[np.argmin(withins)]\n"
_OTSU_START = _OTSU_HEAD + "        ...\n" + _OTSU_TAIL
_SPLIT = "        dark = img[img <= t]\n        light = img[img > t]\n"
_SHARES = "        w_dark = (img <= t).mean()\n        w_light = (img > t).mean()\n"
_OTSU = _OTSU_HEAD + _SPLIT + _SHARES + "        withins.append(w_dark * dark.std() ** 2 + w_light * light.std() ** 2)\n" + _OTSU_TAIL
# The example of exercise 4: a background of grays 2 to 5, with an object of grays 6 to 9 in it.
BLOB = [[2, 3, 3, 4, 3], [3, 8, 9, 5, 2], [4, 9, 8, 6, 3], [3, 7, 3, 2, 4]]

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

EXERCISES = {
    "binary": FunctionBox(
        name="binary",
        picture=yours.arrays(call=0, shade=True),
        function="binary",
        start=_BIN_START,
        solution=_BIN,
        visible=[(SMALL, 100)],
        hidden=[([[100, 101, 99]], 100), ([[0, 0], [255, 255]], 0), ([[12, 250, 7]], 254),
                (np.array([[5, 200, 128, 129]], dtype=np.uint8), 128), ([[60, 70], [80, 90], [100, 110]], 85)],
        show=_listed,
        compare=_same_image,
        first=[_DOTS],
        mistakes=[
            Mistake("true-false", state="almost", matches=lambda got, text: _is_true_false(got),
                    en="`{call}` gives True and False: which pixels are lighter than `t`. The task asks for an image with the grays 255 and 0: multiply by 255. Python counts True as 1 and False as 0.",
                    nl="`{call}` geeft True en False: welke pixels lichter zijn dan `t`. De opdracht vraagt om een afbeelding met de grijstinten 255 en 0: vermenigvuldig met 255. Python telt True als 1 en False als 0."),
            Mistake("ones", state="almost", matches=lambda got, text: _zeros_and_ones(got),
                    en="`{call}` gives `{got}`: 1 for white. In a grayscale image of uint8, white is 255: multiply by 255.",
                    nl="`{call}` geeft `{got}`: 1 voor wit. In een grijswaardenbeeld van uint8 is wit 255: vermenigvuldig met 255."),
            _m("or-equal", "Gray `t` itself becomes white in your image. The lesson puts gray `t` in the dark group: only a gray lighter than `t` becomes white, so compare with `>`.",
               "Grijstint `t` zelf wordt wit in jouw afbeelding. De les zet grijstint `t` in de donkere groep: alleen een grijstint lichter dan `t` wordt wit, dus vergelijk met `>`.",
               wrong=_BIN_HEAD + "    return 255 * (img >= t)"),
            _m("swapped", "Black and white are the other way around: the light pixels should become white, 255, and the dark ones black, 0.",
               "Zwart en wit zijn omgedraaid: de lichte pixels moeten wit worden, 255, en de donkere zwart, 0.",
               wrong=_BIN_HEAD + "    return 255 * (img <= t)"),
            Mistake("fixed-100", state="almost", matches=lambda got, text: re.search(r"\b100\b", text) is not None,
                    en="Your code compares with 100, the threshold of the example. Use `t`: the check also tries other thresholds.",
                    nl="Je code vergelijkt met 100, de drempel van het voorbeeld. Gebruik `t`: de check probeert ook andere drempels."),
            Mistake("minus", state="almost", matches=lambda got, text: re.search(r"img\s*-\s*t\b", text) is not None,
                    en="On an image of uint8, `img - t` wraps around below 0: a pixel darker than `t` then gets a large gray instead of a negative number. Compare the image with `t` itself: `img > t`.",
                    nl="Bij een afbeelding van uint8 loopt `img - t` onder 0 rond: een pixel die donkerder is dan `t` krijgt dan een hoge grijstint in plaats van een negatief getal. Vergelijk de afbeelding met `t` zelf: `img > t`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: 255 for every pixel lighter than `t`, and 0 for the rest.",
                      "nl": "Vervang de `...` door één regel: 255 voor elke pixel die lichter is dan `t`, en 0 voor de rest."},
            "cleared": _CLEARED,
            "arguments": {"en": "The check calls `{call}`, with an image and a threshold, and the `def` line of your function has a different number of names between its parentheses. Keep `(img, t)`.",
                          "nl": "De check roept `{call}` aan, met een afbeelding en een drempel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, t)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Every pixel lighter than `t` becomes 255, and every other pixel 0.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke pixel die lichter is dan `t` wordt 255, en elke andere pixel 0."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function compare every pixel with `t`, whatever `t` is?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijkt je functie elke pixel met `t`, welke drempel `t` ook is?"},
            "correct": {"en": "Your function is right in all {n} cases. A pixel with gray `t` itself stays black, and the image keeps its shape: every pixel gets a new gray of its own.",
                        "nl": "Je functie klopt in alle {n} gevallen. Een pixel met grijstint `t` zelf blijft zwart, en de afbeelding houdt zijn vorm: elke pixel krijgt een eigen nieuwe grijstint."},
        },
    ),
    "lowest": Value(
        name="lowest",
        start=66,
        solution=122,          # cell_threshold(); the test of the lesson measures it again
        outcome=within_at,
        prose_numbers=True,
        mistakes=[
            Mistake("left", matches=lambda got, t: int(t) < cell_threshold(),
                    en="At threshold {value}, the within-group variance is {got}. Further to the right it gets smaller still: move the slider on.",
                    nl="Bij drempel {value} is de variantie binnen de groepen {got}. Verder naar rechts wordt hij nog kleiner: schuif verder."),
            Mistake("right", matches=lambda got, t: int(t) > cell_threshold(),
                    en="At threshold {value}, the within-group variance is {got}. Further to the left it is smaller: move the slider back.",
                    nl="Bij drempel {value} is de variantie binnen de groepen {got}. Verder naar links is hij kleiner: schuif terug."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at 66, where it started: the within-group variance is 470.4 there, and more than half of the pixels are white. Move it to the right.",
                      "nl": "De slider staat nog op 66, waar hij begon: de variantie binnen de groepen is daar 470,4, en meer dan de helft van de pixels is wit. Schuif hem naar rechts."},
            "wrong": {"en": "At threshold {value}, the within-group variance is {got}. Look for the threshold where it is smallest.",
                      "nl": "Bij drempel {value} is de variantie binnen de groepen {got}. Zoek de drempel waar hij het kleinst is."},
            "correct": {"en": "Threshold {value} it is: the within-group variance is {got} there, the smallest. That is Otsu's threshold. The bottom of the curve is flat: from 114 to 130 it hardly changes, and neither does the white disk. And 122 lies almost exactly halfway between the mean gray of the background, 64, and that of the cell, 180.",
                        "nl": "Drempel {value} klopt: de variantie binnen de groepen is daar {got}, de kleinste. Dat is de drempel van Otsu. De bodem van de kromme is vlak: van 114 tot 130 verandert hij nauwelijks, en de witte schijf ook niet. En 122 ligt bijna precies midden tussen de gemiddelde grijstint van de achtergrond, 64, en die van de cel, 180."},
        },
    ),
    "otsu": FunctionBox(
        name="otsu",
        picture=yours.steps(call=0),
        function="otsu",
        start=_OTSU_START,
        solution=_OTSU,
        visible=[(BLOB,)],
        hidden=[([[3, 3, 3, 4, 4, 5], [6, 7, 8, 9, 9, 9]],), ([[2, 2, 3, 3, 3, 4], [4, 4, 5, 6, 7, 8]],),
                ([[0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7]],), ([[250, 251, 252, 253, 254, 255, 255, 255]],),
                ([[4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 9]],), (np.array([[7, 7, 8, 9]], dtype=np.uint8),),
                ([[100, 101, 102, 103], [104, 105, 106, 107]],)],
        first=[_DOTS],
        mistakes=[
            Mistake("less-than", matches=lambda got, text: re.search(r"img\[\s*img\s*<\s*t\s*\]", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`. With `img[img < t]`, gray `t` itself is left out of the dark group, and at the first threshold, the darkest gray, the dark group has no pixels at all. The variance of no pixels is `nan`, not a number, and `np.argmin` picks it. Put gray `t` in the dark group: `img[img <= t]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Met `img[img < t]` valt grijstint `t` zelf buiten de donkere groep, en bij de eerste drempel, de donkerste grijstint, heeft de donkere groep helemaal geen pixels. De variantie van geen pixels is `nan`, geen getal, en `np.argmin` kiest die. Zet grijstint `t` in de donkere groep: `img[img <= t]`."),
            _m("no-weights", "You add up the two variances as they are. Weight each by the share of the pixels in its group: a group of a few pixels then counts for little.",
               "Je telt de twee varianties op zoals ze zijn. Weeg elke variantie met het aandeel van de pixels in zijn groep: een groep van een paar pixels telt dan maar weinig mee.",
               wrong=_OTSU_HEAD + _SPLIT + "        withins.append(dark.std() ** 2 + light.std() ** 2)\n" + _OTSU_TAIL),
            _m("std", "You weight the standard deviations, not the variances. The variance is the square of the standard deviation: `dark.std() ** 2`.",
               "Je weegt de standaardafwijkingen, niet de varianties. De variantie is het kwadraat van de standaardafwijking: `dark.std() ** 2`.",
               wrong=_OTSU_HEAD + _SPLIT + _SHARES + "        withins.append(w_dark * dark.std() + w_light * light.std())\n" + _OTSU_TAIL),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the steps 2 and 3 of the recipe: the dark and the light pixels, their shares, and the within-group variance, added to `withins`.",
                      "nl": "Vervang de `...` door de stappen 2 en 3 van het recept: de donkere en de lichte pixels, hun aandelen, en de variantie binnen de groepen, toegevoegd aan `withins`."},
            "cleared": _CLEARED,
            "arguments": {"en": "The check calls `{call}`, with one image, and the `def` line of your function has a different number of names between its parentheses. Keep `(img)`.",
                          "nl": "De check roept `{call}` aan, met één afbeelding, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For every threshold: the share of the dark pixels times their variance, plus the share of the light pixels times theirs.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Voor elke drempel: het aandeel van de donkere pixels keer hun variantie, plus het aandeel van de lichte pixels keer hun variantie."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does every threshold get its within-group variance, with both groups weighted by their share?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Krijgt elke drempel zijn variantie binnen de groepen, met beide groepen gewogen met hun aandeel?"},
            "correct": {"en": "Your function is right in all {n} cases. On the photo of the cell it gives 122, as in the worked example. scikit-image and OpenCV have this method built in.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op de foto van de cel geeft hij 122, zoals in het uitgewerkte voorbeeld. scikit-image en OpenCV hebben deze methode ingebouwd."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("binary", _BIN_START, "empty", None),
    ("binary", _BIN, "correct", None),
    ("binary", _BIN_HEAD + "    return (img > t).astype(np.uint8) * 255", "correct", None),
    ("binary", _BIN_HEAD + "    new = np.zeros(img.shape)\n    new[img > t] = 255\n    return new", "correct", None),
    ("binary", _BIN_HEAD + "    return img > t", "almost", "true-false"),
    ("binary", _BIN_HEAD + "    return (img > t).astype(int)", "almost", "ones"),
    ("binary", _BIN_HEAD + "    return 255 * (img >= t)", "wrong", "or-equal"),
    ("binary", _BIN_HEAD + "    return 255 * (img <= t)", "wrong", "swapped"),
    ("binary", _BIN_HEAD + "    return 255 * (img < t)", "wrong", None),
    ("binary", _BIN_HEAD + "    return [[0, 255, 255], [0, 255, 0]]", "almost", None),
    ("binary", _BIN_HEAD + "    return 255 * (img > 100)", "almost", "fixed-100"),
    ("binary", _BIN_HEAD + "    return 255 * (img - t > 0)", "almost", "minus"),
    ("binary", _BIN_HEAD + "    ...\n    return 255 * (img > t)", "wrong", "dots-left"),
    ("lowest", 66, "empty", None),
    ("lowest", 122, "correct", None),
    ("lowest", 114, "wrong", "left"),
    ("lowest", 74, "wrong", "left"),
    ("lowest", 130, "wrong", "right"),
    ("lowest", 186, "wrong", "right"),
    ("otsu", _OTSU_START, "empty", None),
    ("otsu", _OTSU, "correct", None),
    ("otsu", _OTSU_HEAD + _SPLIT + "        withins.append(len(dark) * dark.std() ** 2 + len(light) * light.std() ** 2)\n" + _OTSU_TAIL, "correct", None),
    ("otsu", _OTSU_HEAD + _SPLIT + "        within = (((dark - dark.mean()) ** 2).sum() + ((light - light.mean()) ** 2).sum()) / len(img.ravel())\n"
             "        withins.append(within)\n" + _OTSU_TAIL, "correct", None),
    ("otsu", _OTSU_HEAD + _SPLIT + _SHARES + "        withins.append(w_dark * np.var(dark) + w_light * np.var(light))\n" + _OTSU_TAIL, "correct", None),
    ("otsu", _OTSU_HEAD + _SPLIT + "        withins.append(dark.std() ** 2 + light.std() ** 2)\n" + _OTSU_TAIL, "wrong", "no-weights"),
    ("otsu", _OTSU_HEAD + _SPLIT + _SHARES + "        withins.append(w_dark * dark.std() + w_light * light.std())\n" + _OTSU_TAIL, "wrong", "std"),
    ("otsu", _OTSU_HEAD + "        dark = img[img < t]\n        light = img[img >= t]\n"
             "        withins.append((img < t).mean() * dark.std() ** 2 + (img >= t).mean() * light.std() ** 2)\n" + _OTSU_TAIL, "wrong", "less-than"),
    ("otsu", _OTSU_HEAD + _SPLIT + "        withins.append((img < t).mean() * dark.std() ** 2 + (img >= t).mean() * light.std() ** 2)\n" + _OTSU_TAIL, "almost", None),
    ("otsu", _OTSU_HEAD + "        withins.append(img.std() ** 2)\n" + _OTSU_TAIL, "wrong", None),
    ("otsu", _OTSU_HEAD + "        withins.append(abs(t - img.mean()))\n" + _OTSU_TAIL, "almost", None),
    ("otsu", _OTSU_HEAD + "        ...\n        withins.append(0)\n" + _OTSU_TAIL, "wrong", "dots-left"),
]
