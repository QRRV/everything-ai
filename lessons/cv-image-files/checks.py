# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-image-files: how far a number is off on average after a round trip through a file, and
the lowest JPEG quality at which the cat stays close. And the pictures: a corner of the coffee photo three ways, and
the cat's eye at every stop of the slider."""
import io
import re

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, Value
from everything_ai import yours

COFFEE = "data/photos/coffee.png"     # 400 rows, 600 columns, RGB; CC0 (data/README.md)
CAT = "data/photos/chelsea.png"       # 300 rows, 451 columns, RGB; CC0 (data/README.md)
CUP = (slice(60, 92), slice(160, 192))     # 32 by 32 pixels: the white rim of the cup against the red saucer
EYE = (slice(70, 126), slice(140, 196))    # 56 by 56 pixels: the cat's left eye
QUALITIES = [5, 10, 25, 50, 75, 90, 95]    # the stops of the slider
CLOSE = 3                                  # the exercise: a number off by less than this on average


def read(path: str) -> np.ndarray:
    """An image file as an array of uint8, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def work_file(name: str) -> str:
    """For the worked example: the path of a file in the learner's work folder, work/files/cv-image-files/<name>."""
    folder = paths.WORK / "files" / "cv-image-files"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


def round_trip(img: np.ndarray, fmt: str, quality: int = 75) -> tuple[np.ndarray, int]:
    """The image after it was saved as PNG or JPEG and read back, in memory, and the size of the file in bytes."""
    from PIL import Image
    buf = io.BytesIO()
    if fmt == "PNG":
        Image.fromarray(img).save(buf, "PNG")
    else:
        Image.fromarray(img).save(buf, "JPEG", quality=quality)
    data = buf.getvalue()
    return np.asarray(Image.open(io.BytesIO(data))), len(data)


def mean_difference(a, b) -> float:
    """How far a number of `b` is from the same number of `a`, on average."""
    return float(abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).mean())


def changed(a, b) -> float:
    """The share of the numbers that differ."""
    return float((np.asarray(a) != np.asarray(b)).mean())


def grouped(n: int, sep: str) -> str:
    """A whole number with a separator between groups of three digits: 41,606, or 41.606 in Dutch."""
    return f"{n:,}".replace(",", sep)


def road() -> np.ndarray:
    """The road image of exercise 2: 60 rows of 80 pixels, 1 on a road 12 pixels wide that runs diagonally, else 0."""
    img = np.zeros((60, 80), dtype=np.uint8)
    for r in range(60):
        img[r, 10 + r:22 + r] = 1
    return img


def picture(labels: dict[str, str]) -> str:
    """The same corner of the coffee photo three ways, magnified: as it is in the PNG file, after JPEG with quality
    75, and after JPEG with quality 10; with the size of each whole file and how far a number is off on average."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    img = read(COFFEE)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw={"wspace": 0.12})
    for ax, (fmt, q, title) in zip(axes, (("PNG", 75, labels["png"]), ("JPEG", 75, labels["q75"]), ("JPEG", 10, labels["q10"]))):
        back, size = round_trip(img, fmt, q)
        ax.imshow(back[CUP], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(labels["size"].format(n=grouped(size, labels["sep"]),
                                            d=f"{mean_difference(img, back):.1f}".replace(".", labels["dec"])), fontsize=9)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def quality_frame(q: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the cat's eye as it is and after JPEG with quality q, side by side, and what the sentence
    under it says about the whole photo: the size of the file, its share of the PNG, how far a number is off on
    average, and the share of numbers that changed."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    img = read(CAT)
    back, size = round_trip(img, "JPEG", q)
    _, png = round_trip(img, "PNG")
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9), gridspec_kw={"wspace": 0.1})
    for ax, shown, title in zip(axes, (img, back), (labels["png"], labels["jpeg"].format(q=q))):
        ax.imshow(shown[EYE], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    html = plots.render(fig, alt=labels["alt"].format(q=q))
    plt.close(fig)
    return html, {"size": size, "png": png, "share": size / png, "diff": mean_difference(img, back),
                  "changed": changed(img, back)}


def lowest_close_quality() -> int:
    """The lowest quality of the slider at which a number of the cat is off by less than CLOSE on average."""
    img = read(CAT)
    return min(q for q in QUALITIES if mean_difference(img, round_trip(img, "JPEG", q)[0]) < CLOSE)


def _quality_outcome(q):
    """What the slider leads to: how far a number of the cat is off on average at this quality, to two decimals."""
    img = read(CAT)
    return round(mean_difference(img, round_trip(img, "JPEG", int(q))[0]), 2)


# The lesson is done when these are correct. "raw-bytes" and "road-format" are choice questions in the lesson.
REQUIRED = ["raw-bytes", "road-format", "mean-difference", "quality"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_HEAD = ("import numpy as np\n\n\ndef mean_difference(a, b):\n"
         "    a = np.array(a, dtype=np.uint8)\n    b = np.array(b, dtype=np.uint8)\n")
_DIFF_START = _HEAD + "    ...\n"
_DIFF = _HEAD + "    diff = abs(a.astype(int) - b.astype(int))\n    return diff.mean()"
# A 2 by 2 grayscale image, and numbers as a JPEG file might give them back: made up, so that the sum is easy.
BEFORE = [[10, 20], [30, 40]]
AFTER = [[12, 20], [27, 40]]


# Subtracting while the numbers are still uint8: they stay between 0 and 255, so 10 - 12 wraps around to 254.
_WRAPPED = dict(
    en="`{call}` gives `{got}`, and it should give `{want}`. You subtract while the numbers are still uint8: they stay between 0 and 255, so 10 − 12 does not give −2 but 254. Turn both into ordinary whole numbers first, with `.astype(int)`, and subtract after that.",
    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je trekt af terwijl de getallen nog uint8 zijn: die blijven tussen 0 en 255, dus 10 − 12 geeft geen −2 maar 254. Maak van allebei eerst gewone gehele getallen, met `.astype(int)`, en trek daarna af.")


def _subtracts_uint8(got, text: str) -> bool:
    """A wrong answer that subtracts without ever making ordinary numbers of the uint8, such as a loop over the
    pixels: the same wrap-around as `a - b`, written another way."""
    body = text.split("dtype=np.uint8)", 2)[-1]
    return "-" in body and re.search(r"astype|float|\bint\b|int16|int32|int64", body) is None


EXERCISES = {
    "mean-difference": FunctionBox(
        name="mean-difference",
        picture=yours.arrays(call=0, shade=True),
        function="mean_difference",
        start=_DIFF_START,
        solution=_DIFF,
        visible=[(BEFORE, AFTER)],
        hidden=[([[0, 255]], [[255, 0]]), ([[5, 5, 5]], [[5, 5, 5]]), ([[100, 101], [102, 103]], [[103, 102], [101, 100]]),
                ([[[200, 10, 0]], [[0, 0, 0]]], [[[190, 30, 4]], [[1, 0, 2]]])],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("wrapped", wrong=_HEAD + "    return abs(a - b).mean()", **_WRAPPED),
            Mistake("wrapped-ba", wrong=_HEAD + "    return abs(b - a).mean()", **_WRAPPED),
            Mistake("no-abs", wrong=_HEAD + "    diff = a.astype(int) - b.astype(int)\n    return diff.mean()",
                    en="`{call}` gives `{got}`, and it should give `{want}`. A number that went up and one that went down cancel each other out in the mean: 10 became 12, +2, and 30 became 27, −3. Take the size of each difference first, with `abs`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een getal dat omhoogging en een dat omlaagging, heffen elkaar op in het gemiddelde: 10 werd 12, +2, en 30 werd 27, −3. Neem eerst de grootte van elk verschil, met `abs`."),
            Mistake("max", wrong=_HEAD + "    diff = abs(a.astype(int) - b.astype(int))\n    return diff.max()",
                    en="`{call}` gives `{got}`: the biggest difference. The task asks how far a number is off on average: the mean of the differences.",
                    nl="`{call}` geeft `{got}`: het grootste verschil. De opdracht vraagt hoe ver een getal er gemiddeld naast zit: het gemiddelde van de verschillen."),
            Mistake("share", wrong=_HEAD + "    return (a != b).mean()",
                    en="`{call}` gives `{got}`: the share of numbers that changed. The task asks how far they are off on average: the mean of the differences, however big each one is.",
                    nl="`{call}` geeft `{got}`: het deel van de getallen dat veranderde. De opdracht vraagt hoe ver ze er gemiddeld naast zitten: het gemiddelde van de verschillen, hoe groot elk ook is."),
            Mistake("sum", wrong=_HEAD + "    diff = abs(a.astype(int) - b.astype(int))\n    return diff.sum()",
                    en="`{call}` gives `{got}`: all differences added up. Divide by how many numbers there are, or take the mean with `.mean()`.",
                    nl="`{call}` geeft `{got}`: alle verschillen opgeteld. Deel door hoeveel getallen er zijn, of neem het gemiddelde met `.mean()`."),
            Mistake("squared", wrong=_HEAD + "    diff = (a.astype(int) - b.astype(int)) ** 2\n    return diff.mean()",
                    en="`{call}` gives `{got}`: the mean of the squared differences, the mean squared error. The task asks how far a number is off on average, the MAE: take the size of each difference with `abs`, without squaring.",
                    nl="`{call}` geeft `{got}`: het gemiddelde van de gekwadrateerde verschillen, de mean squared error. De opdracht vraagt hoe ver een getal er gemiddeld naast zit, de MAE: neem de grootte van elk verschil met `abs`, zonder te kwadrateren."),
            Mistake("array", matches=lambda got, case: type(got).__name__ == "ndarray",
                    en="`{call}` gives an array: the difference of every number. The task asks for one number, their mean: `.mean()`.",
                    nl="`{call}` geeft een array: het verschil van elk getal. De opdracht vraagt om één getal, hun gemiddelde: `.mean()`."),
            Mistake("int8", matches=lambda got, text: re.search(r"(?<!u)int8", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`. `int8` holds whole numbers from −128 to 127 only, so 255 does not fit in it. Use `int`, which holds any whole number.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. In `int8` passen alleen gehele getallen van −128 tot en met 127, dus 255 past er niet in. Gebruik `int`: daarin past elk geheel getal."),
            Mistake("wrapped-other", matches=_subtracts_uint8, **_WRAPPED),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with how far a number of `b` is from the same number of `a`, on average.",
                      "nl": "Vervang de `...` door hoe ver een getal van `b` gemiddeld van hetzelfde getal van `a` af ligt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two images, and the `def` line of your function has a different number of names between its parentheses. Keep `(a, b)`.",
                          "nl": "De check roept `{call}` aan, met twee afbeeldingen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you turn both into ordinary whole numbers, take the size of every difference, and give back their mean?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Maak je van allebei gewone gehele getallen, neem je de grootte van elk verschil, en geef je hun gemiddelde terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every pair of images?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk paar afbeeldingen?"},
            "correct": {"en": "Your function is right in all {n} cases. As ordinary whole numbers, 10 − 12 is −2, and `abs` makes it 2: a number that went down counts as much as one that went up.",
                        "nl": "Je functie klopt in alle {n} gevallen. Als gewone gehele getallen is 10 − 12 gelijk aan −2, en `abs` maakt er 2 van: een getal dat omlaagging, telt even zwaar als een dat omhoogging."},
        },
    ),
    "quality": Value(
        name="quality",
        start=95,
        solution=75,          # lowest_close_quality(); the test of the lesson measures it again
        outcome=_quality_outcome,
        expect_from=lambda: _quality_outcome(lowest_close_quality()),
        compare=lambda got, want: abs(got - want) < 1e-9,
        prose_numbers=True,
        mistakes=[
            Mistake("higher", state="almost", matches=lambda got, q: got < CLOSE,
                    en="With quality {value}, a number is off by {got} on average: less than 3, so the cat stays close. Can the quality go lower, for a smaller file, while a number is still off by less than 3?",
                    nl="Met kwaliteit {value} zit een getal er gemiddeld {got} naast: minder dan 3, dus de kat blijft dichtbij. Kan de kwaliteit lager, voor een kleiner bestand, terwijl een getal er nog minder dan 3 naast zit?"),
            Mistake("too-low", matches=lambda got, q: got >= CLOSE,
                    en="With quality {value}, a number is off by {got} on average: 3 or more. Move the slider up to a higher quality.",
                    nl="Met kwaliteit {value} zit een getal er gemiddeld {got} naast: 3 of meer. Zet de slider op een hogere kwaliteit."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at quality 95, where it started. Move it down, and watch how far a number is off on average.",
                      "nl": "De slider staat nog op kwaliteit 95, waar hij begon. Schuif hem omlaag, en kijk hoe ver een getal er gemiddeld naast zit."},
            "wrong": {"en": "With quality {value}, a number is off by {got} on average. Look for the lowest quality with less than 3.",
                      "nl": "Met kwaliteit {value} zit een getal er gemiddeld {got} naast. Zoek de laagste kwaliteit met minder dan 3."},
            "correct": {"en": "Quality {value} it is: a number is off by {got} on average, and one step lower it is 3 or more. The file is a twentieth of the numbers, and you hardly see a difference.",
                        "nl": "Kwaliteit {value} is het: een getal zit er gemiddeld {got} naast, en één stap lager is dat 3 of meer. Het bestand is een twintigste van de getallen, en je ziet amper verschil."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("mean-difference", _DIFF_START, "empty", None),
    ("mean-difference", _DIFF, "correct", None),
    ("mean-difference", _HEAD + "    return abs(b.astype(int) - a.astype(int)).mean()", "correct", None),
    ("mean-difference", _HEAD + "    a = a.astype(int)\n    b = b.astype(int)\n    return float(np.mean(abs(a - b)))", "correct", None),
    ("mean-difference", _HEAD + "    diff = abs(a / 1 - b / 1)\n    return diff.mean()", "correct", None),
    ("mean-difference", _HEAD + "    return abs(a - b).mean()", "wrong", "wrapped"),
    ("mean-difference", _HEAD + "    return abs(b - a).mean()", "wrong", "wrapped-ba"),
    ("mean-difference", _HEAD + "    return (a - b).mean()", "wrong", "wrapped"),
    ("mean-difference", _HEAD + "    return (a.astype(int) - b.astype(int)).mean()", "wrong", "no-abs"),
    ("mean-difference", _HEAD + "    return abs(a.astype(int) - b.astype(int)).max()", "wrong", "max"),
    ("mean-difference", _HEAD + "    return (a != b).mean()", "wrong", "share"),
    ("mean-difference", _HEAD + "    return abs(a.astype(int) - b.astype(int)).sum()", "wrong", "sum"),
    ("mean-difference", _HEAD + "    return abs(a.astype(int) - b.astype(int))", "wrong", "array"),
    ("mean-difference", _HEAD + "    return ((a.astype(int) - b.astype(int)) ** 2).mean()", "wrong", "squared"),
    ("mean-difference", _HEAD + "    a = a.astype(np.int8)\n    b = b.astype(np.int8)\n    return abs(a - b).mean()", "wrong", "int8"),
    ("mean-difference", _HEAD + "    total = 0\n    for x, y in zip(a.flatten(), b.flatten()):\n        total = total + abs(x - y)\n"
                                "    return total / a.size", "wrong", "wrapped-other"),
    ("mean-difference", _HEAD + "    total = 0\n    for x, y in zip(a.flatten(), b.flatten()):\n        total = total + abs(int(x) - int(y))\n"
                                "    return total / a.size", "correct", None),
    ("mean-difference", _HEAD + "    return 1.25", "almost", None),
    ("quality", 95, "empty", None),
    ("quality", 75, "correct", None),
    ("quality", 90, "almost", "higher"),
    ("quality", 50, "wrong", "too-low"),
    ("quality", 5, "wrong", "too-low"),
]
