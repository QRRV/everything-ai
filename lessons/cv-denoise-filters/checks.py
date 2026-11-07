# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-denoise-filters: why the weights of the Gaussian filter are divided by 16, the mean filter,
the window size of the mean filter that brings the grainy photo closest to the clean one, and the median filter.
And the pictures: a part of the photo of the coffee cup with grain and with specks, each next to what the mean, the
Gaussian and the median filter make of it; and for the slider, the grainy part after a mean filter of one size next to
how far the result is from the clean photo for every size."""
import functools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value

COFFEE = "data/photos/coffee.png"       # 600 by 400 pixels, RGB; from the sample data of scikit-image
PART = (slice(0, 300), slice(100, 500))  # the part with the cup: rows 0 to 300, columns 100 to 500
SEED, SIGMA, SPECKS = 0, 25, 0.05       # the seed, the standard deviation of the grain, the share of black specks
GAUSS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
SIZES = list(range(1, 16, 2))           # the stops of the slider: windows of 1 by 1 up to 15 by 15
ZOOM = (slice(120, 220), slice(230, 360))  # the part the pictures show: the rim of the cup, the saucer, the spoon


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def make_noise(clean, seed=SEED, sigma=SIGMA, specks=SPECKS) -> tuple[np.ndarray, np.ndarray]:
    """Grain and specks, made the way the worked example makes them: first the grain, then the specks, from one
    random generator with a seed."""
    rng = np.random.default_rng(seed)
    grainy = clean + rng.normal(0, sigma, size=clean.shape)
    u = rng.random(size=clean.shape)
    speckled = np.array(clean)
    speckled[u < specks] = 0
    speckled[u > 1 - specks] = 255
    return grainy, speckled


@functools.lru_cache(maxsize=1)
def _photos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = read(COFFEE).max(axis=2)[PART]
    grainy, speckled = make_noise(clean)
    for a in (clean, grainy, speckled):
        a.flags.writeable = False
    return clean, grainy, speckled


def clean_photo() -> np.ndarray:
    """The photo of the lesson: the part of the coffee photo with the cup, in grays (the value of HSV)."""
    return _photos()[0]


def grainy_photo() -> np.ndarray:
    return _photos()[1]


def speckled_photo() -> np.ndarray:
    return _photos()[2]


def _windows(img, k: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(np.asarray(img, dtype=float), (k, k))


def mean_of(img, k: int = 3) -> np.ndarray:
    """The mean filter of k by k, only where the whole window fits: the result is k - 1 smaller in each direction."""
    return _windows(img, k).mean(axis=(2, 3))


def gauss_of(img) -> np.ndarray:
    """The Gaussian filter of the lesson, 3 by 3, only where the whole window fits."""
    return (_windows(img, 3) * GAUSS).sum(axis=(2, 3))


def median_of(img, k: int = 3) -> np.ndarray:
    """The median filter of k by k, only where the whole window fits."""
    return np.median(_windows(img, k), axis=(2, 3))


def off(out, clean) -> float:
    """How far a filtered image is from the clean photo, on average, compared with the clean photo without the
    pixels the filter left off at the border."""
    clean = np.asarray(clean, dtype=float)
    b = (clean.shape[0] - out.shape[0]) // 2
    return float(np.abs(out - clean[b:clean.shape[0] - b, b:clean.shape[1] - b]).mean())


@functools.lru_cache(maxsize=1)
def _size_offs() -> tuple[float, ...]:
    """How far the grainy photo is from the clean one after a mean filter of every size of the slider, worked out
    once: the slider draws a picture for every stop."""
    return tuple(off(mean_of(grainy_photo(), k), clean_photo()) for k in SIZES)


def best_size() -> int:
    """The window size with the smallest difference: the answer of exercise 3."""
    return SIZES[int(np.argmin(_size_offs()))]


def off_at(k) -> float:
    """What the slider leads to: how far the grainy photo is from the clean one after a mean filter of k by k, to one
    decimal, as the sentence under the picture shows it."""
    return round(_size_offs()[SIZES.index(int(k))], 1)


def _zoom(out, k: int = 3) -> np.ndarray:
    """The part of a filtered image the pictures show; the result of a filter of k by k starts k // 2 further in."""
    b = k // 2
    rows, cols = ZOOM
    return out[rows.start - b:rows.stop - b, cols.start - b:cols.stop - b]


def _photo_axes(ax, img, title):
    ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def filter_picture(labels: dict[str, str]) -> str:
    """A part of the photo of the cup: at the top with grain, at the bottom with specks; each next to what the mean,
    the Gaussian and the median filter of 3 by 3 make of it."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), dpi=110, gridspec_kw={"hspace": 0.22, "wspace": 0.06})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    for row, (key, noisy) in enumerate([("grainy", grainy_photo()), ("speckled", speckled_photo())]):
        _photo_axes(axes[row, 0], _zoom(np.asarray(noisy, dtype=float), 1), labels[key])
        _photo_axes(axes[row, 1], _zoom(mean_of(noisy)), labels["mean"])
        _photo_axes(axes[row, 2], _zoom(gauss_of(noisy)), labels["gauss"])
        _photo_axes(axes[row, 3], _zoom(median_of(noisy)), labels["median"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def size_frame(k: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the grainy part after a mean filter of k by k, and next to it how far the result is from the
    clean photo for every size, with k marked. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    k = int(k)
    offs = _size_offs()
    here = off_at(k)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.0, 1.3], "wspace": 0.3})
    _photo_axes(left, _zoom(mean_of(grainy_photo(), k), k), labels["window"].format(k=k))
    right.plot(SIZES, offs, color=plots.LOSS, linewidth=1.8, marker="o", markersize=3.5)
    right.plot([k, k], [0, offs[SIZES.index(k)]], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([k], [offs[SIZES.index(k)]], marker="o", color=plots.HYPER, markersize=8)
    right.set_xlim(0, 16)
    right.set_ylim(0, 22)
    right.set_xticks(SIZES)
    right.set_xlabel(labels["size"])
    right.set_title(labels["off"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(k=k))
    plt.close(fig)
    return html, {"off": here}


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _listed(v) -> str:
    """An image the way the lesson writes it: an array as a list of rows, on one line, without needless decimals."""
    a = _numbers(v)
    if a is None or a.ndim != 2:
        return str(v.tolist() if isinstance(v, np.ndarray) else v)
    rows = [[int(x) if float(x).is_integer() else round(float(x), 2) for x in row] for row in a]
    return str(rows)


def _same_image(got, want) -> bool:
    """Right when every pixel has the course's gray, up to rounding, in the same shape."""
    a = _numbers(got)
    return a is not None and a.shape == np.shape(want) and bool(np.allclose(a, np.asarray(want, dtype=float), atol=1e-9))


def _all_zeros(got) -> bool:
    a = _numbers(got)
    return a is not None and a.size > 0 and bool(np.all(a == 0))


def _never_filled(got, text) -> bool:
    """Only zeros back, and no line that stores something in `out`."""
    return _all_zeros(got) and re.search(r"^[ \t]*out\[", text, re.MULTILINE) is None


def _has_nan(got) -> bool:
    a = _numbers(got)
    return a is not None and bool(np.isnan(a).any())


def _past_out(e) -> bool:
    """An index past the end of `out`, because the loops go over `img`, which is 2 bigger in each direction."""
    if not isinstance(e, IndexError):
        return False
    tb = e.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    here = tb.tb_frame.f_locals if tb is not None else {}
    out, r, c = here.get("out"), here.get("r"), here.get("c")
    try:
        return isinstance(r, int) and (r >= len(out) or isinstance(c, int) and c >= len(out[0]))
    except TypeError:
        return False


# The lesson is done when these are correct. "weights" is a choice question in the lesson.
REQUIRED = ["weights", "mean", "size", "median"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_CENTERED = re.compile(r"img\[\s*r\s*-\s*1")


def _centered(got, text) -> bool:
    """A window that starts one row and column before r, c: the first one starts at row -1, and is empty."""
    return _CENTERED.search(text) is not None and _has_nan(got)

_MEAN_HEAD = ("import numpy as np\n\n\ndef mean_filter(img):\n    img = np.array(img, dtype=float)\n"
              "    out = np.zeros((len(img) - 2, len(img[0]) - 2))\n"
              "    for r in range(len(out)):\n        for c in range(len(out[0])):\n")
_MEAN_TAIL = "    return out\n"
_MEAN_START = _MEAN_HEAD + "            ...\n" + _MEAN_TAIL
_MEAN = _MEAN_HEAD + "            out[r, c] = img[r:r + 3, c:c + 3].mean()\n" + _MEAN_TAIL
# The example of exercise 2: an edge from 0 to 9, four rows of five pixels.
STEP = [[0, 0, 0, 9, 9], [0, 0, 0, 9, 9], [0, 0, 0, 9, 9], [0, 0, 0, 9, 9]]

_MED_HEAD = ("import numpy as np\n\n\ndef median_filter(img):\n    img = np.array(img, dtype=float)\n"
             "    out = np.zeros((len(img) - 2, len(img[0]) - 2))\n")
_MED_TAIL = "    return out\n"
_MED_START = _MED_HEAD + "    ...\n" + _MED_TAIL
_LOOPS = "    for r in range(len(out)):\n        for c in range(len(out[0])):\n"
_MED = _MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r:r + 3, c:c + 3])\n" + _MED_TAIL
# The example of exercise 4: a white speck at the top left, and an edge between gray 10 and gray 200.
SPECK_EDGE = [[10, 10, 10, 200, 200], [10, 255, 10, 200, 200], [10, 10, 10, 200, 200], [10, 10, 10, 200, 200]]

_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

EXERCISES = {
    "mean": FunctionBox(
        name="mean",
        function="mean_filter",
        start=_MEAN_START,
        solution=_MEAN,
        visible=[(STEP,)],
        hidden=[([[10, 10, 10], [10, 100, 10], [10, 10, 10]],), ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],),
                ([[0, 9, 0], [9, 0, 9], [0, 9, 0], [9, 0, 9]],),
                (np.array([[250, 250, 250, 250], [250, 250, 250, 250], [250, 250, 250, 250]], dtype=np.uint8),),
                ([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3], [5, 8, 9, 7, 9], [3, 2, 3, 8, 4], [6, 2, 6, 4, 3]],)],
        show=_listed,
        compare=_same_image,
        first=[_DOTS],
        mistakes=[
            Mistake("centered", matches=_centered,
                    en="`{call}` gives `{got}`, and it should give `{want}`. `out[r, c]` belongs to the window that starts at row `r` and column `c`: `img[r:r + 3, c:c + 3]`. With `r - 1`, the first window starts at row -1, and `img[-1:2]` has no rows: Python counts -1 from the end.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `out[r, c]` hoort bij het venster dat begint bij rij `r` en kolom `c`: `img[r:r + 3, c:c + 3]`. Met `r - 1` begint het eerste venster bij rij -1, en `img[-1:2]` heeft geen rijen: Python telt -1 vanaf het eind."),
            Mistake("past-out", error=_past_out,
                    en="With `{call}`, your function stops with an error: your loops go past the end of `out`. `out` is 2 smaller than `img` in each direction, so go over the places of `out`: `for r in range(len(out)):`, and inside it `for c in range(len(out[0])):`.",
                    nl="Met `{call}` stopt je functie met een fout: je loops gaan voorbij het eind van `out`. `out` is in elke richting 2 kleiner dan `img`, dus ga langs de plekken van `out`: `for r in range(len(out)):`, en daarbinnen `for c in range(len(out[0])):`."),
            Mistake("many", error=lambda e: isinstance(e, ValueError) and "sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `out[r, c]` is one place, and it gets the whole window of nine grays. Store one number there: the mean of the window, with `.mean()`.",
                    nl="Met `{call}` stopt je functie met een fout: `out[r, c]` is één plek, en die krijgt het hele venster van negen grijstinten. Sla er één getal in op: het gemiddelde van het venster, met `.mean()`."),
            _m("chained", "`img[r:r + 3][c:c + 3]` takes rows twice: the second `[c:c + 3]` picks rows from the three rows the first one took, not columns. Rows and columns go in one pair of brackets, with a comma: `img[r:r + 3, c:c + 3]`.",
               "`img[r:r + 3][c:c + 3]` neemt twee keer rijen: de tweede `[c:c + 3]` kiest rijen uit de drie rijen die de eerste nam, geen kolommen. Rijen en kolommen staan tussen één paar haken, met een komma: `img[r:r + 3, c:c + 3]`.",
               wrong=_MEAN_HEAD + "            out[r, c] = img[r:r + 3][c:c + 3].mean()\n" + _MEAN_TAIL),
            Mistake("zeros", matches=_never_filled,
                    en="`{call}` gives only zeros: `out` is still the array of zeros it started as. Store the mean of each window in it: `out[r, c] = ...`.",
                    nl="`{call}` geeft alleen nullen: `out` is nog steeds de array met nullen van het begin. Sla het gemiddelde van elk venster erin op: `out[r, c] = ...`."),
            _m("sum", "You add up the nine grays. The mean divides that sum by 9: `.mean()`, or `.sum() / 9`.",
               "Je telt de negen grijstinten op. Het gemiddelde deelt die som door 9: `.mean()`, of `.sum() / 9`.",
               wrong=_MEAN_HEAD + "            out[r, c] = img[r:r + 3, c:c + 3].sum()\n" + _MEAN_TAIL),
            _m("two", "Your window is 2 by 2: `r:r + 2` takes rows `r` and `r + 1`. A window of 3 by 3 is `img[r:r + 3, c:c + 3]`.",
               "Je venster is 2 bij 2: `r:r + 2` neemt de rijen `r` en `r + 1`. Een venster van 3 bij 3 is `img[r:r + 3, c:c + 3]`.",
               wrong=_MEAN_HEAD + "            out[r, c] = img[r:r + 2, c:c + 2].mean()\n" + _MEAN_TAIL),
            _m("copy", "You copy one pixel: `img[r, c]`. The mean filter takes the mean of the whole window of 3 by 3 that starts there: `img[r:r + 3, c:c + 3]`.",
               "Je kopieert één pixel: `img[r, c]`. Het gemiddeldefilter neemt het gemiddelde van het hele venster van 3 bij 3 dat daar begint: `img[r:r + 3, c:c + 3]`.",
               wrong=_MEAN_HEAD + "            out[r, c] = img[r, c]\n" + _MEAN_TAIL),
            _m("median", "That is the median of each window: the median filter of exercise 4. The mean filter takes the mean: `.mean()`.",
               "Dat is de mediaan van elk venster: het mediaanfilter van oefening 4. Het gemiddeldefilter neemt het gemiddelde: `.mean()`.",
               wrong=_MEAN_HEAD + "            out[r, c] = np.median(img[r:r + 3, c:c + 3])\n" + _MEAN_TAIL),
            _m("whole", "Every pixel gets the mean of the whole image. Take the mean of the window of 3 by 3 only: `img[r:r + 3, c:c + 3]`.",
               "Elke pixel krijgt het gemiddelde van de hele afbeelding. Neem alleen het gemiddelde van het venster van 3 bij 3: `img[r:r + 3, c:c + 3]`.",
               wrong=_MEAN_HEAD + "            out[r, c] = img.mean()\n" + _MEAN_TAIL),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: the mean of the window of 3 by 3 that starts at row `r` and column `c`, stored in `out[r, c]`.",
                      "nl": "Vervang de `...` door één regel: het gemiddelde van het venster van 3 bij 3 dat begint bij rij `r` en kolom `c`, opgeslagen in `out[r, c]`."},
            "cleared": _CLEARED,
            "arguments": {"en": "The check calls `{call}`, with one image, and the `def` line of your function has a different number of names between its parentheses. Keep `(img)`.",
                          "nl": "De check roept `{call}` aan, met één afbeelding, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Every place `out[r, c]` gets the mean of the nine grays of `img[r:r + 3, c:c + 3]`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke plek `out[r, c]` krijgt het gemiddelde van de negen grijstinten van `img[r:r + 3, c:c + 3]`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does every place get the mean of its own window of 3 by 3, whatever the image?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Krijgt elke plek het gemiddelde van zijn eigen venster van 3 bij 3, welke afbeelding het ook is?"},
            "correct": {"en": "Your function is right in all {n} cases. The edge from 0 to 9 became 0, 3, 6: one step became two smaller ones. That is the blur.",
                        "nl": "Je functie klopt in alle {n} gevallen. De rand van 0 naar 9 werd 0, 3, 6: één stap werd twee kleinere. Dat is de vervaging."},
        },
    ),
    "size": Value(
        name="size",
        start=1,
        solution=5,            # best_size(); the test of the lesson measures it again
        outcome=off_at,
        prose_numbers=True,
        mistakes=[
            Mistake("small", matches=lambda got, k: int(k) < best_size(),
                    en="With a window of {value} by {value}, the filtered photo is on average {got} grays off. A bigger window cancels more of the grain: move the slider further right.",
                    nl="Met een venster van {value} bij {value} zit de gefilterde foto gemiddeld {got} grijstinten ernaast. Een groter venster middelt meer korrel weg: schuif verder naar rechts."),
            Mistake("big", matches=lambda got, k: int(k) > best_size(),
                    en="With a window of {value} by {value}, the filtered photo is on average {got} grays off. The window is so big that the blur at the edges costs more than the grain it removes: move the slider back.",
                    nl="Met een venster van {value} bij {value} zit de gefilterde foto gemiddeld {got} grijstinten ernaast. Het venster is zo groot dat het vervagen van de randen meer kost dan het weghalen van korrel oplevert: schuif terug."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at 1, where it started: a window of 1 by 1 is the pixel itself, so nothing is filtered, and the photo is on average 20.0 grays off. Move it to the right.",
                      "nl": "De slider staat nog op 1, waar hij begon: een venster van 1 bij 1 is de pixel zelf, dus er wordt niets gefilterd, en de foto zit gemiddeld 20,0 grijstinten ernaast. Schuif hem naar rechts."},
            "wrong": {"en": "With a window of {value} by {value}, the filtered photo is on average {got} grays off. Look for the size where that is smallest.",
                      "nl": "Met een venster van {value} bij {value} zit de gefilterde foto gemiddeld {got} grijstinten ernaast. Zoek de grootte waar dat het kleinst is."},
            "correct": {"en": "A window of {value} by {value} is right: the filtered photo is on average {got} grays off, the least of all sizes. With 3 by 3 the grain still shows, and from 7 by 7 on, the blur at the edges costs more than the extra grain it removes. With stronger grain, the best window is bigger: with a standard deviation of 40, it is 7 by 7.",
                        "nl": "Een venster van {value} bij {value} klopt: de gefilterde foto zit gemiddeld {got} grijstinten ernaast, het minst van alle groottes. Met 3 bij 3 is de korrel nog te zien, en vanaf 7 bij 7 kost het vervagen van de randen meer dan het weghalen van de extra korrel oplevert. Bij sterkere korrel is het beste venster groter: met een standaardafwijking van 40 is het 7 bij 7."},
        },
    ),
    "median": FunctionBox(
        name="median",
        function="median_filter",
        start=_MED_START,
        solution=_MED,
        visible=[(SPECK_EDGE,)],
        hidden=[([[10, 10, 10], [10, 100, 10], [10, 10, 10]],), ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],),
                ([[0, 9, 0], [9, 0, 9], [0, 9, 0], [9, 0, 9]],),
                (np.array([[0, 255, 0, 255], [255, 0, 255, 0], [0, 255, 0, 255]], dtype=np.uint8),),
                ([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3], [5, 8, 9, 7, 9], [3, 2, 3, 8, 4], [6, 2, 6, 4, 3]],)],
        show=_listed,
        compare=_same_image,
        first=[_DOTS],
        mistakes=[
            Mistake("centered", matches=_centered,
                    en="`{call}` gives `{got}`, and it should give `{want}`. `out[r, c]` belongs to the window that starts at row `r` and column `c`: `img[r:r + 3, c:c + 3]`. With `r - 1`, the first window starts at row -1, and `img[-1:2]` has no rows: Python counts -1 from the end.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `out[r, c]` hoort bij het venster dat begint bij rij `r` en kolom `c`: `img[r:r + 3, c:c + 3]`. Met `r - 1` begint het eerste venster bij rij -1, en `img[-1:2]` heeft geen rijen: Python telt -1 vanaf het eind."),
            Mistake("past-out", error=_past_out,
                    en="With `{call}`, your function stops with an error: your loops go past the end of `out`. `out` is 2 smaller than `img` in each direction, so go over the places of `out`: `for r in range(len(out)):`, and inside it `for c in range(len(out[0])):`.",
                    nl="Met `{call}` stopt je functie met een fout: je loops gaan voorbij het eind van `out`. `out` is in elke richting 2 kleiner dan `img`, dus ga langs de plekken van `out`: `for r in range(len(out)):`, en daarbinnen `for c in range(len(out[0])):`."),
            Mistake("many", error=lambda e: isinstance(e, ValueError) and "sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `out[r, c]` is one place, and it gets the whole window of nine grays. Store one number there: the median of the window, with `np.median(...)`.",
                    nl="Met `{call}` stopt je functie met een fout: `out[r, c]` is één plek, en die krijgt het hele venster van negen grijstinten. Sla er één getal in op: de mediaan van het venster, met `np.median(...)`."),
            _m("chained", "`img[r:r + 3][c:c + 3]` takes rows twice: the second `[c:c + 3]` picks rows from the three rows the first one took, not columns. Rows and columns go in one pair of brackets, with a comma: `img[r:r + 3, c:c + 3]`.",
               "`img[r:r + 3][c:c + 3]` neemt twee keer rijen: de tweede `[c:c + 3]` kiest rijen uit de drie rijen die de eerste nam, geen kolommen. Rijen en kolommen staan tussen één paar haken, met een komma: `img[r:r + 3, c:c + 3]`.",
               wrong=_MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r:r + 3][c:c + 3])\n" + _MED_TAIL),
            Mistake("zeros", matches=_never_filled,
                    en="`{call}` gives only zeros: `out` is still the array of zeros it started as. Fill every place of it, in two loops as in exercise 2: `for r in range(len(out)):`, and inside it `for c in range(len(out[0])):`.",
                    nl="`{call}` geeft alleen nullen: `out` is nog steeds de array met nullen van het begin. Vul elke plek ervan, in twee loops zoals in oefening 2: `for r in range(len(out)):`, en daarbinnen `for c in range(len(out[0])):`."),
            _m("mean", "That is the mean of each window: the mean filter of exercise 2. The median filter takes the median: `np.median(...)`.",
               "Dat is het gemiddelde van elk venster: het gemiddeldefilter van oefening 2. Het mediaanfilter neemt de mediaan: `np.median(...)`.",
               wrong=_MED_HEAD + _LOOPS + "            out[r, c] = img[r:r + 3, c:c + 3].mean()\n" + _MED_TAIL),
            _m("two", "Your window is 2 by 2: `r:r + 2` takes rows `r` and `r + 1`. A window of 3 by 3 is `img[r:r + 3, c:c + 3]`.",
               "Je venster is 2 bij 2: `r:r + 2` neemt de rijen `r` en `r + 1`. Een venster van 3 bij 3 is `img[r:r + 3, c:c + 3]`.",
               wrong=_MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r:r + 2, c:c + 2])\n" + _MED_TAIL),
            _m("whole", "Every pixel gets the median of the whole image. Take the median of the window of 3 by 3 only: `img[r:r + 3, c:c + 3]`.",
               "Elke pixel krijgt de mediaan van de hele afbeelding. Neem alleen de mediaan van het venster van 3 bij 3: `img[r:r + 3, c:c + 3]`.",
               wrong=_MED_HEAD + _LOOPS + "            out[r, c] = np.median(img)\n" + _MED_TAIL),
            _m("copy", "You copy one pixel: `img[r, c]`. The median filter takes the median of the whole window of 3 by 3 that starts there: `img[r:r + 3, c:c + 3]`.",
               "Je kopieert één pixel: `img[r, c]`. Het mediaanfilter neemt de mediaan van het hele venster van 3 bij 3 dat daar begint: `img[r:r + 3, c:c + 3]`.",
               wrong=_MED_HEAD + _LOOPS + "            out[r, c] = img[r, c]\n" + _MED_TAIL),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two loops over every place of `out`, as in exercise 2, and fill each place with the median of its window of 3 by 3.",
                      "nl": "Vervang de `...` door twee loops over elke plek van `out`, zoals in oefening 2, en vul elke plek met de mediaan van zijn venster van 3 bij 3."},
            "cleared": _CLEARED,
            "arguments": {"en": "The check calls `{call}`, with one image, and the `def` line of your function has a different number of names between its parentheses. Keep `(img)`.",
                          "nl": "De check roept `{call}` aan, met één afbeelding, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Every place `out[r, c]` gets the median of the nine grays of `img[r:r + 3, c:c + 3]`. A 0 where another number belongs may be a place your loops never reached.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke plek `out[r, c]` krijgt de mediaan van de negen grijstinten van `img[r:r + 3, c:c + 3]`. Een 0 waar een ander getal hoort, kan een plek zijn waar je loops nooit kwamen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do your loops reach every place of `out`, and does every place get the median of its own window of 3 by 3?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Komen je loops bij elke plek van `out`, en krijgt elke plek de mediaan van zijn eigen venster van 3 bij 3?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example the speck is gone, and the edge between 10 and 200 is still one step: 10, 10, 200.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld is de spikkel weg, en de rand tussen 10 en 200 is nog steeds één stap: 10, 10, 200."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_W = "            window = img[r:r + 3, c:c + 3]\n"
CASES = [
    ("mean", _MEAN_START, "empty", None),
    ("mean", _MEAN, "correct", None),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r:r + 3, c:c + 3].sum() / 9\n" + _MEAN_TAIL, "correct", None),
    ("mean", _MEAN_HEAD + "            out[r, c] = (img[r:r + 3, c:c + 3] * np.ones((3, 3)) / 9).sum()\n" + _MEAN_TAIL, "correct", None),
    ("mean", _MEAN_HEAD + _W + "            out[r, c] = np.mean(window)\n" + _MEAN_TAIL, "correct", None),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r - 1:r + 2, c - 1:c + 2].mean()\n" + _MEAN_TAIL, "wrong", "centered"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r:r + 3, c:c + 3].sum()\n" + _MEAN_TAIL, "wrong", "sum"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r:r + 2, c:c + 2].mean()\n" + _MEAN_TAIL, "wrong", "two"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r, c]\n" + _MEAN_TAIL, "wrong", "copy"),
    ("mean", _MEAN_HEAD + "            out[r, c] = np.median(img[r:r + 3, c:c + 3])\n" + _MEAN_TAIL, "wrong", "median"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img.mean()\n" + _MEAN_TAIL, "wrong", "whole"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r:r + 3, c:c + 3].mean() // 1\n" + _MEAN_TAIL, "almost", None),
    ("mean", _MEAN_HEAD + "            ...\n            out[r, c] = 0\n" + _MEAN_TAIL, "wrong", "dots-left"),
    ("mean", _MEAN_HEAD + "            img[r:r + 3, c:c + 3].mean()\n" + _MEAN_TAIL, "wrong", "zeros"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r:r + 3][c:c + 3].mean()\n" + _MEAN_TAIL, "wrong", "chained"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r:r + 3, c:c + 3]\n" + _MEAN_TAIL, "error", "many"),
    ("mean", _MEAN_HEAD.replace("range(len(out))", "range(len(img))").replace("range(len(out[0]))", "range(len(img[0]))")
     + "            out[r, c] = img[r:r + 3, c:c + 3].mean()\n" + _MEAN_TAIL, "error", "past-out"),
    ("mean", _MEAN_HEAD + "            out[r, c] = img[r + 3, c]\n" + _MEAN_TAIL, "error", None),
    ("size", 1, "empty", None),
    ("size", 5, "correct", None),
    ("size", 3, "wrong", "small"),
    ("size", 7, "wrong", "big"),
    ("size", 15, "wrong", "big"),
    ("median", _MED_START, "empty", None),
    ("median", _MED, "correct", None),
    ("median", _MED_HEAD + _LOOPS + _W + "            out[r, c] = np.sort(window.ravel())[4]\n" + _MED_TAIL, "correct", None),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r - 1:r + 2, c - 1:c + 2])\n" + _MED_TAIL, "wrong", "centered"),
    ("median", _MED_HEAD + "    pass\n" + _MED_TAIL, "wrong", "zeros"),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = img[r:r + 3, c:c + 3].mean()\n" + _MED_TAIL, "wrong", "mean"),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r:r + 2, c:c + 2])\n" + _MED_TAIL, "wrong", "two"),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = np.median(img)\n" + _MED_TAIL, "wrong", "whole"),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = img[r, c]\n" + _MED_TAIL, "wrong", "copy"),
    ("median", _MED_HEAD + "    for r in range(len(out)):\n        out[r, 0] = np.median(img[r:r + 3, 0:3])\n" + _MED_TAIL, "wrong", None),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r:r + 3, c + 1])\n" + _MED_TAIL, "almost", None),
    ("median", _MED_HEAD + "    ...\n    out[0, 0] = 1\n" + _MED_TAIL, "wrong", "dots-left"),
    ("median", _MED_HEAD + "    for r in range(len(img)):\n        for c in range(len(img[0])):\n"
     "            out[r, c] = np.median(img[r:r + 3, c:c + 3])\n" + _MED_TAIL, "error", "past-out"),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = img[r:r + 3, c:c + 3]\n" + _MED_TAIL, "error", "many"),
    ("median", _MED_HEAD + _LOOPS + "            out[r, c] = np.median(img[r:r + 3][c:c + 3])\n" + _MED_TAIL, "wrong", "chained"),
    ("median", _MED_HEAD + "    for r in range(1, len(img) - 1):\n        for c in range(1, len(img[0]) - 1):\n"
     "            out[r - 1, c - 1] = np.median(img[r - 1:r + 2, c - 1:c + 2])\n" + _MED_TAIL, "correct", None),
    ("median", _MED_HEAD + "    for r in range(1, len(img) - 1):\n        for c in range(1, len(img[0]) - 1):\n"
     "            out[r - 1, c - 1] = np.median(img[r - 1:r + 2, c - 1:c + 1])\n" + _MED_TAIL, "wrong", None),
]
