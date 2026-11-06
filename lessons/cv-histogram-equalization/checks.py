# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-histogram-equalization: where the grays of a photo of snow end up, the cumulative
histogram as shares, the gray of the dull photo that becomes about the middle of the scale, and histogram equalization
itself. And the pictures: the dull photo and the photo after equalization, each with its histogram and its
cumulative histogram, and the cumulative histogram of the dull photo with one gray marked, for the slider."""
import functools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value

CAT = "data/photos/chelsea.png"     # 451 by 300 pixels, RGB; CC0, from the sample data of scikit-image
GRAYS = list(range(130, 161))       # the stops of the slider: the grays in the middle of the dull photo


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def gray_photo() -> np.ndarray:
    """The photo of the cat in grays, as the worked example makes it: the value of HSV of every pixel, the
    strongest of its red, green and blue."""
    return read(CAT).max(axis=2)


def dull_photo() -> np.ndarray:
    """The photo of the lesson made dull on purpose, as the worked example does: every gray g becomes
    100 + 0.3 g, as through fog."""
    return (100 + 0.3 * gray_photo()).astype(np.uint8)


def counts_of(img) -> np.ndarray:
    """How many pixels have each gray, from 0 to 255."""
    return np.bincount(np.asarray(img, dtype=np.int64).ravel(), minlength=256)


def shares_of(img) -> np.ndarray:
    """The cumulative histogram as shares: for every gray, the share of the pixels with that gray or a darker one."""
    counts = counts_of(img)
    return np.cumsum(counts) / counts.sum()


def table_of(img) -> np.ndarray:
    """The new gray of every gray: 255 times its share, as uint8."""
    return (255 * shares_of(img)).astype(np.uint8)


def equalized(img) -> np.ndarray:
    """Histogram equalization, as the lesson does it."""
    img = np.asarray(img)
    return table_of(img)[img]


@functools.lru_cache(maxsize=1)
def _dull_shares() -> np.ndarray:
    """The shares of the dull photo, worked out once: the slider draws a picture for every stop."""
    shares = shares_of(dull_photo())
    shares.flags.writeable = False
    return shares


def middle_gray() -> int:
    """The darkest gray of the dull photo that becomes 128 or lighter: the answer of exercise 3."""
    return int(np.argmax((255 * _dull_shares()).astype(np.uint8) >= 128))


def _new_gray(g) -> int:
    """What the slider leads to: the gray that gray g of the dull photo becomes."""
    return int((255 * _dull_shares()).astype(np.uint8)[int(g)])


def _photo_axes(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def before_after(labels: dict[str, str]) -> str:
    """The dull photo and the photo after equalization, each with its histogram and its cumulative histogram, on
    the scale from 0 to 255; a dashed line in the cumulative histograms is the straight line from 0 to 1."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    dull = dull_photo()
    after = equalized(dull)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), dpi=160,
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0], "hspace": 0.45, "wspace": 0.3})
    for row, (img, name) in enumerate([(dull, "dull"), (after, "after")]):
        _photo_axes(axes[row, 0], img, labels[name])
        ax = axes[row, 1]
        ax.bar(range(256), counts_of(img), width=1.0, color=plots.DATA)
        ax.set_xlim(-3, 258)
        ax.set_xticks([0, 128, 255])
        ax.set_yticks([])
        ax.set_title(labels["histogram"], fontsize=10)
        ax = axes[row, 2]
        ax.plot([0, 255], [0, 1], color=plots.DATA, linestyle="--", linewidth=1.0)
        ax.plot(range(256), shares_of(img), color=plots.INK, linewidth=1.8, drawstyle="steps-post")
        ax.set_xlim(-3, 258)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([0, 128, 255])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_title(labels["cumulative"], fontsize=10)
    for ax in axes[1, 1:]:
        ax.set_xlabel(labels["gray"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def gray_frame(g: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the cumulative histogram of the dull photo as shares, with gray g marked, and on the right
    the same curve read as the new gray, 255 times the share. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    shares = _dull_shares()
    share = float(shares[g])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    shown = range(96, 175)
    ax.plot(shown, shares[96:175], color=plots.INK, linewidth=1.8, drawstyle="steps-post")
    ax.plot([g, g], [0, share], color=plots.INK_3, linestyle=":", linewidth=1.2)
    ax.plot([g, 174], [share, share], color=plots.INK_3, linestyle=":", linewidth=1.2)
    ax.plot([g], [share], marker="o", color=plots.INK, markersize=7)
    ax.set_xlim(96, 174)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(labels["gray"])
    ax.set_ylabel(labels["share"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    right = ax.secondary_yaxis("right", functions=(lambda s: 255 * s, lambda n: n / 255))
    right.set_yticks([0, 64, 128, 192, 255])
    right.set_ylabel(labels["new"])
    html = plots.render(fig, alt=labels["alt"].format(g=g))
    plt.close(fig)
    return html, {"share": share, "new": _new_gray(g)}


def _rounded(v) -> str:
    """A list or array of numbers as the lesson writes it: at most six decimals, so that 0.3333333333333333 shows
    as 0.333333 and two different numbers never look the same; an array as a list."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if x != x:
            return "nan"
        if float(x).is_integer() and isinstance(v, (int, np.integer)):
            return str(int(x))
        text = f"{x:.6f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_rounded(x) for x in v) + "]"
    return repr(v)


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    try:
        a = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None
    return a


def _last_is_100(got) -> bool:
    a = _numbers(got)
    return a is not None and a.ndim == 1 and a.size > 0 and abs(a[-1] - 100) < 1e-6


def _same_grays(got, want) -> bool:
    """Exercise 4 counts as right when every pixel gets a gray, a whole number from 0 to 255, within 1 of the
    course's: the course cuts off the decimals with astype, and rounding them is just as good."""
    a = _numbers(got)
    if a is None or a.shape != np.shape(want):
        return False
    return bool(np.all(a == np.floor(a)) and np.all((a >= 0) & (a <= 255))
                and np.all(np.abs(a - np.asarray(want, dtype=float)) <= 1))


def _rounded_shares(got, text) -> bool:
    """Shares that were rounded on purpose: the code rounds, and what it gives still climbs from dark to light up
    to about 1, as shares do."""
    a = _numbers(got)
    if a is None or a.ndim != 1 or a.size == 0 or re.search(r"\bround\b|:\.\d", text) is None:
        return False
    return bool(np.all(np.diff(a) >= 0) and abs(a[-1] - 1) < 0.01)


def _has_decimals(got) -> bool:
    a = _numbers(got)
    return a is not None and a.size > 0 and bool(np.any(a != np.floor(a)))


def _is_table(got) -> bool:
    return np.ndim(got) == 1 and np.size(got) == 256


# The lesson is done when these are correct. "snow" is a choice question in the lesson.
REQUIRED = ["snow", "cumulative", "middle", "equalize"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CUM_HEAD = "import numpy as np\n\n\ndef cumulative(counts):\n    counts = np.array(counts)\n"
_CUM_START = _CUM_HEAD + "    ...\n"
_CUM = _CUM_HEAD + "    return np.cumsum(counts) / counts.sum()"

_EQ_HEAD = ("import numpy as np\n\n\ndef equalize(img):\n    img = np.array(img)\n"
            "    counts = np.array([(img == g).sum() for g in range(256)])\n")
_EQ_START = _EQ_HEAD + "    ...\n"
_EQ = _EQ_HEAD + ("    share = np.cumsum(counts) / counts.sum()\n"
                  "    table = (255 * share).astype(np.uint8)\n"
                  "    return table[img]")
# The example of exercise 4: six pixels, two of gray 100, three of gray 120 and one of gray 200.
SMALL = [[100, 100, 120], [120, 120, 200]]

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


EXERCISES = {
    "cumulative": FunctionBox(
        name="cumulative",
        function="cumulative",
        start=_CUM_START,
        solution=_CUM,
        visible=[([2, 0, 3, 1],)],
        hidden=[([5],), ([0, 0, 4],), ([1, 1, 1, 1],), ([10, 0, 0, 30, 60],), ([0, 7, 0, 0, 1, 0],)],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            _m("not-divided", "Those are numbers of pixels: the running total. A share says which part of all the pixels that is: divide by how many pixels there are, `counts.sum()`.",
               "Dat zijn aantallen pixels: het lopende totaal. Een aandeel zegt welk deel van alle pixels dat is: deel door het aantal pixels, `counts.sum()`.",
               wrong=_CUM_HEAD + "    return np.cumsum(counts)"),
            _m("no-cumsum", "That is the share of each gray by itself. The cumulative histogram counts every gray together with all the darker ones: add up first, with `np.cumsum`, then divide.",
               "Dat is het aandeel van elke grijstint op zich. Het cumulatieve histogram telt elke grijstint samen met alle donkerdere: tel eerst op, met `np.cumsum`, en deel dan.",
               wrong=_CUM_HEAD + "    return counts / counts.sum()"),
            _m("by-length", "You divide by `len(counts)`, the number of grays. Divide by the number of pixels, `counts.sum()`: then the lightest gray gets share 1.",
               "Je deelt door `len(counts)`, het aantal grijstinten. Deel door het aantal pixels, `counts.sum()`: dan krijgt de lichtste grijstint aandeel 1.",
               wrong=_CUM_HEAD + "    return np.cumsum(counts) / len(counts)"),
            _m("darker-only", "Your shares count only the darker pixels, not the pixels of the gray itself, so the lightest gray does not get 1. `np.cumsum` already counts the gray itself too.",
               "Je aandelen tellen alleen de donkerdere pixels, niet de pixels van de grijstint zelf, dus de lichtste grijstint krijgt geen 1. `np.cumsum` telt de grijstint zelf al mee.",
               wrong=_CUM_HEAD + "    return (np.cumsum(counts) - counts) / counts.sum()"),
            _m("lighter", "Your shares go down instead of up: they count the gray and the lighter ones. The cumulative histogram adds up from dark to light, in the order of `counts`.",
               "Je aandelen gaan omlaag in plaats van omhoog: ze tellen de grijstint en de lichtere. Het cumulatieve histogram telt op van donker naar licht, in de volgorde van `counts`.",
               wrong=_CUM_HEAD + "    return np.cumsum(counts[::-1])[::-1] / counts.sum()"),
            Mistake("percent", state="almost", matches=lambda got, text: _last_is_100(got),
                    en="`{call}` gives `{got}`: percentages, with 100 for the lightest gray. The task asks for shares, from 0 to 1: leave out the `100 *`.",
                    nl="`{call}` geeft `{got}`: percentages, met 100 voor de lichtste grijstint. De opdracht vraagt om aandelen, van 0 tot en met 1: laat de `100 *` weg."),
            Mistake("rounded", state="almost", matches=_rounded_shares,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Do not round: the check compares the exact shares, and only shows them with six decimals.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Rond niet af: de check vergelijkt de precieze aandelen, en laat ze alleen zien met zes decimalen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the share of every gray: the running total of `counts`, divided by the number of pixels.",
                      "nl": "Vervang de `...` door het aandeel van elke grijstint: het lopende totaal van `counts`, gedeeld door het aantal pixels."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of counts, and the `def` line of your function has a different number of names between its parentheses. Keep `(counts)`.",
                          "nl": "De check roept `{call}` aan, met één lijst met aantallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(counts)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For every gray: the pixels of that gray and of all the darker ones, divided by all the pixels.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Voor elke grijstint: de pixels van die grijstint en van alle donkerdere, gedeeld door alle pixels."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every list of counts, also one with a single gray, or with zeros at the start?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lijst met aantallen, ook een met één grijstint, of met nullen aan het begin?"},
            "correct": {"en": "Your function is right in all {n} cases. The shares only go up, and the lightest gray always gets 1: all the pixels have that gray or a darker one.",
                        "nl": "Je functie klopt in alle {n} gevallen. De aandelen gaan alleen omhoog, en de lichtste grijstint krijgt altijd 1: alle pixels hebben die grijstint of een donkerdere."},
        },
    ),
    "middle": Value(
        name="middle",
        start=130,
        solution=145,          # middle_gray(); the test of the lesson measures it again
        outcome=_new_gray,
        expect_from=lambda: _new_gray(middle_gray()),
        compare=lambda got, want: got == want,
        mistakes=[
            Mistake("too-dark", matches=lambda got, g: got < 128,
                    en="Gray {value} becomes gray {got}: still darker than 128. Move the slider to the right.",
                    nl="Grijstint {value} wordt grijstint {got}: nog donkerder dan 128. Schuif de slider naar rechts."),
            Mistake("lighter", state="almost", matches=lambda got, g: int(g) > middle_gray(),
                    en="Gray {value} becomes gray {got}: 128 or lighter, yes. But a darker gray of the dull photo does too. Which is the darkest?",
                    nl="Grijstint {value} wordt grijstint {got}: 128 of lichter, ja. Maar een donkerdere grijstint van de doffe foto ook. Welke is de donkerste?"),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at 130, where it started: that gray becomes 20. Move it to the right, and watch the share climb.",
                      "nl": "De slider staat nog op 130, waar hij begon: die grijstint wordt 20. Schuif hem naar rechts, en kijk hoe het aandeel stijgt."},
            "wrong": {"en": "Gray {value} becomes gray {got}. Look for the darkest gray that becomes 128 or lighter.",
                      "nl": "Grijstint {value} wordt grijstint {got}. Zoek de donkerste grijstint die 128 of lichter wordt."},
            "correct": {"en": "Gray {value} it is: it becomes gray {got}, and gray 144 only becomes 119. Just over half of the pixels have gray 145 or a darker one, so 145 is the median of the dull photo: equalization puts the median near the middle of the scale.",
                        "nl": "Het is grijstint {value}: die wordt grijstint {got}, en grijstint 144 wordt maar 119. Net iets meer dan de helft van de pixels heeft grijstint 145 of een donkerdere, dus 145 is de mediaan van de doffe foto: egalisatie zet de mediaan vlak bij het midden van de schaal."},
        },
    ),
    "equalize": FunctionBox(
        name="equalize",
        function="equalize",
        start=_EQ_START,
        solution=_EQ,
        visible=[(SMALL,)],
        hidden=[([[0, 255]],), ([[50, 50], [50, 50]],), ([[10, 20, 30, 40]],),
                ([[3, 3, 3, 200], [3, 90, 90, 255], [3, 3, 90, 200]],),
                (np.array([[7, 7, 8, 9, 7, 8]], dtype=np.uint8),)],
        show=_rounded,
        compare=_same_grays,
        first=[_DOTS],
        mistakes=[
            Mistake("table", matches=lambda got, text: _is_table(got),
                    en="`{call}` gives 256 numbers: the table, one new gray for every gray. Give back the image itself, with every pixel looked up in the table: `table[img]`.",
                    nl="`{call}` geeft 256 getallen: de tabel, één nieuwe grijstint voor elke grijstint. Geef de afbeelding zelf terug, met elke pixel opgezocht in de tabel: `table[img]`."),
            _m("by-256", "You multiply by 256. The lightest gray has share 1, so it gets 256, one more than a gray of uint8 can be: as uint8 it wraps around to 0, black. Multiply by 255, the lightest gray there is.",
               "Je vermenigvuldigt met 256. De lichtste grijstint heeft aandeel 1, dus die krijgt 256, één meer dan een grijstint van uint8 kan zijn: als uint8 begint hij weer bij 0, zwart. Vermenigvuldig met 255, de lichtste grijstint die er is.",
               matches=lambda got, text: re.search(r"256\s*\*|\*\s*256", text) is not None),
            _m("no-cumsum", "Every gray now gets 255 times its own share, so most pixels turn dark. Use the share of the gray together with all the darker ones: `np.cumsum(counts)`.",
               "Elke grijstint krijgt nu 255 keer zijn eigen aandeel, dus de meeste pixels worden donker. Gebruik het aandeel van de grijstint samen met alle donkerdere: `np.cumsum(counts)`.",
               wrong=_EQ_HEAD + "    share = counts / counts.sum()\n    table = (255 * share).astype(np.uint8)\n    return table[img]"),
            _m("not-divided", "The running total counts pixels, and 255 times that goes far past 255, so uint8 wraps around. Divide it by the number of pixels first: then the shares run from 0 to 1.",
               "Het lopende totaal telt pixels, en 255 keer dat gaat ver voorbij 255, dus uint8 begint weer bij 0. Deel het eerst door het aantal pixels: dan lopen de aandelen van 0 tot en met 1.",
               wrong=_EQ_HEAD + "    table = (255 * np.cumsum(counts)).astype(np.uint8)\n    return table[img]"),
            _m("darker-only", "Your shares count only the darker pixels, not the pixels of the gray itself, so the lightest gray does not become 255. `np.cumsum` already counts the gray itself too.",
               "Je aandelen tellen alleen de donkerdere pixels, niet de pixels van de grijstint zelf, dus de lichtste grijstint wordt geen 255. `np.cumsum` telt de grijstint zelf al mee.",
               wrong=_EQ_HEAD + "    share = (np.cumsum(counts) - counts) / counts.sum()\n    table = (255 * share).astype(np.uint8)\n    return table[img]"),
            Mistake("decimals", state="almost", matches=lambda got, text: _has_decimals(got),
                    en="`{call}` gives `{got}`: grays with decimals. A pixel of uint8 is a whole number: make the table whole numbers with `.astype(np.uint8)`, before you look up the pixels.",
                    nl="`{call}` geeft `{got}`: grijstinten met decimalen. Een pixel van uint8 is een geheel getal: maak gehele getallen van de tabel met `.astype(np.uint8)`, voordat je de pixels opzoekt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with three steps: the share of every gray, the table of new grays, and every pixel looked up in the table.",
                      "nl": "Vervang de `...` door drie stappen: het aandeel van elke grijstint, de tabel met nieuwe grijstinten, en elke pixel opgezocht in de tabel."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one image, and the `def` line of your function has a different number of names between its parentheses. Keep `(img)`.",
                          "nl": "De check roept `{call}` aan, met één afbeelding, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Every pixel gets 255 times the share of the pixels with its gray or a darker one.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke pixel krijgt 255 keer het aandeel van de pixels met zijn grijstint of een donkerdere."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every image, also one with a single gray, or with black and white in it?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke afbeelding, ook een met één grijstint, of met zwart en wit erin?"},
            "correct": {"en": "Your function is right in all {n} cases. An image with only one gray becomes all white: every pixel has share 1. On the dull photo, it gives the photo of the worked example.",
                        "nl": "Je functie klopt in alle {n} gevallen. Een afbeelding met maar één grijstint wordt helemaal wit: elke pixel heeft aandeel 1. Op de doffe foto geeft hij de foto van het uitgewerkte voorbeeld."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cumulative", _CUM_START, "empty", None),
    ("cumulative", _CUM, "correct", None),
    ("cumulative", _CUM_HEAD + "    total = np.cumsum(counts)\n    return total / total[-1]", "correct", None),
    ("cumulative", _CUM_HEAD + "    shares = []\n    seen = 0\n    for c in counts:\n        seen = seen + c\n        shares.append(seen / counts.sum())\n    return shares", "correct", None),
    ("cumulative", _CUM_HEAD + "    return np.cumsum(counts)", "wrong", "not-divided"),
    ("cumulative", _CUM_HEAD + "    return counts / counts.sum()", "wrong", "no-cumsum"),
    ("cumulative", _CUM_HEAD + "    return np.cumsum(counts) / len(counts)", "wrong", "by-length"),
    ("cumulative", _CUM_HEAD + "    return (np.cumsum(counts) - counts) / counts.sum()", "wrong", "darker-only"),
    ("cumulative", _CUM_HEAD + "    return np.cumsum(counts[::-1])[::-1] / counts.sum()", "wrong", "lighter"),
    ("cumulative", _CUM_HEAD + "    return 100 * np.cumsum(counts) / counts.sum()", "almost", "percent"),
    ("cumulative", _CUM_HEAD + "    return [1 / 3, 1 / 3, 5 / 6, 1.0]", "almost", None),
    ("cumulative", _CUM_HEAD + "    return np.round(np.cumsum(counts) / counts.sum(), 3)", "almost", "rounded"),
    ("cumulative", _CUM_HEAD + "    return [round(x, 4) for x in np.cumsum(counts) / counts.sum()]", "almost", "rounded"),
    ("cumulative", _CUM_HEAD + "    return np.cumsum(counts / counts.sum())", "correct", None),
    ("middle", 130, "empty", None),
    ("middle", 145, "correct", None),
    ("middle", 144, "wrong", "too-dark"),
    ("middle", 131, "wrong", "too-dark"),
    ("middle", 146, "almost", "lighter"),
    ("middle", 160, "almost", "lighter"),
    ("equalize", _EQ_START, "empty", None),
    ("equalize", _EQ, "correct", None),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / counts.sum()\n    table = np.round(255 * share).astype(int)\n    return table[img]", "correct", None),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / img.size\n    new = np.zeros(img.shape, dtype=np.uint8)\n"
                 "    for g in range(256):\n        new[img == g] = int(255 * share[g])\n    return new", "correct", None),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / counts.sum()\n    return (255 * share).astype(np.uint8)", "wrong", "table"),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / counts.sum()\n    table = (256 * share).astype(np.uint8)\n    return table[img]", "wrong", "by-256"),
    ("equalize", _EQ_HEAD + "    share = counts / counts.sum()\n    table = (255 * share).astype(np.uint8)\n    return table[img]", "wrong", "no-cumsum"),
    ("equalize", _EQ_HEAD + "    table = (255 * np.cumsum(counts)).astype(np.uint8)\n    return table[img]", "wrong", "not-divided"),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / counts.sum()\n    table = 255 * share\n    return table[img]", "almost", "decimals"),
    ("equalize", _EQ_HEAD + "    return np.array([[85, 85, 212], [212, 212, 255]], dtype=np.uint8)", "almost", None),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / counts.sum()\n    table = (256 * share).astype(int)\n    return table[img]", "wrong", "by-256"),
    ("equalize", _EQ_HEAD + "    share = (np.cumsum(counts) - counts) / counts.sum()\n    table = (255 * share).astype(np.uint8)\n    return table[img]", "wrong", "darker-only"),
    ("equalize", _EQ_HEAD + "    share = np.cumsum(counts) / counts.sum()\n    table = np.ceil(255 * share).astype(np.uint8)\n    return table[img]", "correct", None),
]
