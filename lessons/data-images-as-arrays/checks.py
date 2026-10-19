# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-images-as-arrays: the sun cut out of a small landscape, and the number of pixels
with much red in a color image. And the pixels of the picture."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

IMAGE = "lessons/data-images-as-arrays/landscape.png"   # made for the course: 24 rows, 40 columns, RGB

# The picture: a grayscale plus sign, 5 by 5; 0 is black, 1 is white.
PLUS = [[1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.5, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0]]


def picture(labels: dict[str, str]) -> tuple[str, str]:
    """The plus sign twice: as the numbers of its array, and as the image those numbers make."""
    import matplotlib.pyplot as plt
    from everything_ai import draw, plots
    numbers = draw.array_grids([(labels["array"], [[f"{x:g}" for x in row] for row in PLUS], ["shape (5, 5)"])], indices=True)
    fig = plt.figure(figsize=(2.2, 2.2))
    plt.imshow(PLUS, cmap="gray", vmin=0, vmax=1)
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.title(labels["image"])
    image = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return numbers.text, image


# The lesson is done when these are correct. "color-shape" is the choice question in the lesson.
REQUIRED = ["color-shape", "sun", "red-pixels"]

_READ = f'img = plt.imread("{IMAGE}")'
_SUN_START = "import matplotlib.pyplot as plt\n\n" + _READ + "\nsun = ...\nsun\n"
_SUN = "import matplotlib.pyplot as plt\n\n" + _READ + "\nsun = img[2:10, 29:37]\nsun"


def _shape(*shape):
    return lambda got, text: type(got).__name__ == "ndarray" and got.shape == shape


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_RED_START = "import numpy as np\n\n\ndef red_pixels(rgb, limit):\n    img = np.array(rgb)\n    ...\n"
_RED = ("import numpy as np\n\n\ndef red_pixels(rgb, limit):\n    img = np.array(rgb)\n"
        "    red = img[:, :, 0]\n    return len(red[red > limit])")
# A color image of 2 rows of 3 pixels: black, gray-blue and orange; orange, red and dark blue.
EXAMPLE = [[[0.0, 0.0, 0.0], [0.5, 0.5, 0.8], [0.8, 0.2, 0.0]],
           [[1.0, 0.5, 0.0], [1.0, 0.2, 0.2], [0.2, 0.2, 0.5]]]


def _red(new: str) -> str:
    """The solution with its last two lines replaced."""
    return _RED.replace("    red = img[:, :, 0]\n    return len(red[red > limit])", new)


EXERCISES = {
    "sun": CodeBox(
        name="sun",
        start=_SUN_START,
        solution=_SUN,
        keep=["import matplotlib.pyplot as plt", _READ],
        first=[_DOTS],
        mistakes=[
            Mistake("too-many-parts", error=lambda e: isinstance(e, IndexError) and "too many indices" in str(e),
                    en="Your index has more parts than the image has directions: three, the rows, the columns and the colors, separated by commas. A slice uses a colon: `2:10`, not `2,10`.",
                    nl="Je index heeft meer delen dan de afbeelding richtingen heeft: drie, de rijen, de kolommen en de kleuren, gescheiden door komma's. Een slice gebruikt een dubbele punt: `2:10`, niet `2,10`."),
            Mistake("swapped", matches=lambda got, text: _shape(0, 8, 3)(got, text) and re.search(r"img\[\s*29\s*:", text) is not None,
                    en="Your part is empty: rows 29 to 36 do not exist, because the image has only 24 rows. The rows come first, then the columns: `img[rows, columns]`.",
                    nl="Je stuk is leeg: de rijen 29 tot en met 36 bestaan niet, want de afbeelding heeft maar 24 rijen. De rijen komen eerst, dan de kolommen: `img[rijen, kolommen]`."),
            Mistake("chained", matches=_shape(0, 40, 3),
                    en="Your part is empty. With two pairs of square brackets, `img[2:10][29:37]`, the second slice works on the rows again: of the 8 rows you took, it asks for rows 29 to 36. Put both slices in one pair, separated by a comma: `img[rows, columns]`.",
                    nl="Je stuk is leeg. Met twee paar vierkante haken, `img[2:10][29:37]`, werkt de tweede slice weer op de rijen: van de 8 rijen die je nam, vraagt hij rij 29 tot en met 36. Zet beide slices in één paar haken, met een komma ertussen: `img[rijen, kolommen]`."),
            Mistake("one-short", matches=_shape(7, 7, 3),
                    en="Your part is 7 by 7 pixels, and the sun is 8 by 8. A slice stops just before its second number: to include row 9 and column 36, stop at 10 and 37.",
                    nl="Je stuk is 7 bij 7 pixels, en de zon is 8 bij 8. Een slice stopt net voor zijn tweede getal: om rij 9 en kolom 36 mee te nemen, stop je bij 10 en 37."),
            Mistake("one-channel", matches=_shape(8, 8),
                    en="Your part has the right pixels, but only one channel: its shape is `(8, 8)`. Take all three colors: leave out the third part of the index.",
                    nl="Je stuk heeft de goede pixels, maar het heeft maar één channel: zijn shape is `(8, 8)`. Neem alle drie de kleuren: laat het derde deel van de index weg."),
            Mistake("shown-not-kept", matches=lambda got, text: type(got).__name__ == "AxesImage" or got is None and _last_line(text).startswith("plt."),
                    en="Your last line draws the image, but the check needs the array itself. Make the last line just `sun`. To see the sun too, put `plt.imshow(sun)` and `plt.show()` above that last line.",
                    nl="Je laatste regel tekent de afbeelding, maar de check heeft de array zelf nodig. Maak de laatste regel gewoon `sun`. Wil je de zon ook zien, zet dan `plt.imshow(sun)` en `plt.show()` boven die laatste regel."),
            Mistake("shape-only", matches=lambda got, text: isinstance(got, tuple) and got == (8, 8, 3),
                    en="That is the shape of your part, `(8, 8, 3)`, and it is the right one. But the check needs the part itself: make the last line just `sun`.",
                    nl="Dat is de shape van je stuk, `(8, 8, 3)`, en die klopt. Maar de check heeft het stuk zelf nodig: maak de laatste regel gewoon `sun`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the part of `img` with the sun: rows 2 to 9, columns 29 to 36.",
                      "nl": "Vervang de `...` door het stuk van `img` met de zon: rij 2 tot en met 9, kolom 29 tot en met 36."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `sun` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `sun` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your part is not the sun yet. It should have shape `(8, 8, 3)`: rows 2 to 9 and columns 29 to 36, with all three colors. Do the rows come first, and does each slice stop one past the last row or column you want?",
                      "nl": "Je stuk is nog niet de zon. Het moet shape `(8, 8, 3)` hebben: rij 2 tot en met 9 en kolom 29 tot en met 36, met alle drie de kleuren. Komen de rijen eerst, en stopt elke slice één voorbij de laatste rij of kolom die je wilt?"},
            "correct": {"en": "That is the sun: 8 rows, 8 columns and 3 colors, shape `(8, 8, 3)`. Its middle pixel is mostly red and green, which together make yellow.",
                        "nl": "Dat is de zon: 8 rijen, 8 kolommen en 3 kleuren, shape `(8, 8, 3)`. Zijn middelste pixel is vooral rood en groen, en samen maken die geel."},
        },
    ),
    "red-pixels": FunctionBox(
        name="red-pixels",
        function="red_pixels",
        start=_RED_START,
        solution=_RED,
        visible=[(EXAMPLE, 0.5)],
        hidden=[([[[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]]], 0.5), ([[[1.0, 0.0, 0.0]]], 0.5),
                ([[[0.7, 0.9, 0.9], [0.8, 0.0, 0.0]]], 0.7),
                ([[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0]], [[0.9, 0.9, 0.9], [0.3, 0.0, 0.0]]], 0.35)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("if-array", error=lambda e: isinstance(e, ValueError) and "truth value of an array" in str(e),
                    en="With `{call}`, your code asks `if` about an array. A comparison with an array gives `True` or `False` for every number in it, and `if` needs one. Use the mask between square brackets instead: `red[red > limit]`, and count those with `len`.",
                    nl="Met `{call}` vraagt je code `if` over een array. Een vergelijking met een array geeft `True` of `False` voor elk getal erin, en `if` heeft er één nodig. Zet het mask liever tussen vierkante haken: `red[red > limit]`, en tel die met `len`."),
            Mistake("green", wrong=_red("    red = img[:, :, 1]\n    return len(red[red > limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count the green channel. The colors go red, green, blue, and the count starts at 0: red is `img[:, :, 0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt het groene channel. De kleuren gaan rood, groen, blauw, en de telling begint bij 0: rood is `img[:, :, 0]`."),
            Mistake("blue", wrong=_red("    red = img[:, :, 2]\n    return len(red[red > limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count the blue channel. The colors go red, green, blue, and the count starts at 0: red is `img[:, :, 0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt het blauwe channel. De kleuren gaan rood, groen, blauw, en de telling begint bij 0: rood is `img[:, :, 0]`."),
            Mistake("all-channels", wrong=_red("    return len(img[img > limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your mask is on the whole image, so it counts every color, also green and blue. Take the red channel first: `img[:, :, 0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je mask staat op de hele afbeelding, dus telt hij elke kleur, ook groen en blauw. Neem eerst het rode channel: `img[:, :, 0]`."),
            Mistake("column", wrong=_red("    red = img[:, 0]\n    return len(red[red > limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `img[:, 0]` is column 0 of every row, with all three colors. The color is the third part of the index: `img[:, :, 0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `img[:, 0]` is kolom 0 van elke rij, met alle drie de kleuren. De kleur is het derde deel van de index: `img[:, :, 0]`."),
            Mistake("row", wrong=_red("    red = img[0]\n    return len(red[red > limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `img[0]` is row 0, with all its pixels and colors. The color is the third part of the index: `img[:, :, 0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `img[0]` is rij 0, met al zijn pixels en kleuren. De kleur is het derde deel van de index: `img[:, :, 0]`."),
            Mistake("less", wrong=_red("    red = img[:, :, 0]\n    return len(red[red < limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count the pixels with less red than `limit`. The task asks for more: use `>`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de pixels met minder rood dan `limit`. De opdracht vraagt om meer: gebruik `>`."),
            Mistake("at-least", wrong=_red("    red = img[:, :, 0]\n    return len(red[red >= limit])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you also count the pixels with exactly as much red as `limit`. The task asks for more: use `>`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt ook de pixels met precies zoveel rood als `limit`. De opdracht vraagt om meer: gebruik `>`."),
            Mistake("fixed-limit", state="almost", wrong=_red("    red = img[:, :, 0]\n    return len(red[red > 0.5])"),
                    en="Right for the example. But `{call}` gives `{got}`, and it should give `{want}`: your mask compares with `0.5`, not with `limit`. Use the name `limit`, so the function works for every limit.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: je mask vergelijkt met `0.5`, niet met `limit`. Gebruik de naam `limit`, dan werkt de functie voor elke grens."),
            Mistake("rows-not-pixels", wrong=_red("    red = img[:, :, 0]\n    return len(red)"),
                    en="`{call}` gives `{got}`: the number of rows of the image. Keep the pixels with more red than `limit` first, with a mask, and count those.",
                    nl="`{call}` geeft `{got}`: het aantal rijen van de afbeelding. Houd eerst de pixels met meer rood dan `limit` over, met een mask, en tel die."),
            Mistake("pixels-not-count", wrong=_red("    red = img[:, :, 0]\n    return red[red > limit]"),
                    en="`{call}` gives `{got}`: the red values themselves. The task asks how many there are: count them with `len`.",
                    nl="`{call}` geeft `{got}`: de roodwaarden zelf. De opdracht vraagt hoeveel het er zijn: tel ze met `len`."),
            Mistake("mask-itself", wrong=_red("    red = img[:, :, 0]\n    return red > limit"),
                    en="`{call}` gives the mask itself: `True` or `False` for every pixel. Put it between the square brackets of `red` to keep those pixels, and count them.",
                    nl="`{call}` geeft het mask zelf: `True` of `False` voor elke pixel. Zet het tussen de vierkante haken van `red` om die pixels over te houden, en tel ze."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the number of pixels of `img` that have more red than `limit`.",
                      "nl": "Vervang de `...` door het aantal pixels van `img` met meer rood dan `limit`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with an image and a limit, and the `def` line of your function has a different number of names between its parentheses. Keep `(rgb, limit)`.",
                          "nl": "De check roept `{call}` aan, met een afbeelding en een grens, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rgb, limit)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the red channel, keep its values above `limit` with a mask, and give back how many there are?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je het rode channel, houd je met een mask de waarden boven `limit` over, en geef je terug hoeveel het er zijn?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every image and every limit, also when no pixel has enough red?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke afbeelding en elke grens, ook als geen pixel genoeg rood heeft?"},
            "correct": {"en": "Your function is right in all {n} cases. One channel is a 2D array of its own, and a mask on it picks pixels from the whole image.",
                        "nl": "Je functie klopt in alle {n} gevallen. Eén channel is een eigen 2D-array, en een mask erop kiest pixels uit de hele afbeelding."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_S = "import matplotlib.pyplot as plt\n\n" + _READ + "\n"
_R = "import numpy as np\n\n\ndef red_pixels(rgb, limit):\n    img = np.array(rgb)\n"
CASES = [
    ("sun", _SUN_START, "empty", None),
    ("sun", _SUN, "correct", None),
    ("sun", _S + "sun = img[2:10, 29:37, :]\nsun", "correct", None),
    ("sun", _S + "sun = img[29:37, 2:10]\nsun", "wrong", "swapped"),
    ("sun", _S + "sun = img[2:9, 29:36]\nsun", "wrong", "one-short"),
    ("sun", _S + "sun = img[2:10, 29:37, 0]\nsun", "wrong", "one-channel"),
    ("sun", _S + "sun = img[2:10, 29:37]\nplt.imshow(sun)", "wrong", "shown-not-kept"),
    ("sun", _S + "sun = img[2:10, 29:37]\nprint(sun)", "wrong", "print-last"),
    ("sun", _S + "sun = img[0:8, 0:8]\nsun", "wrong", None),
    ("sun", _S + "sun = img[2:10, 29:37]\nplt.imshow(sun)\nplt.show()", "wrong", "shown-not-kept"),
    ("sun", _S + "sun = img[2:10, 29:37]\nsun.shape", "wrong", "shape-only"),
    ("sun", _S + "sun = img[2:10][29:37]\nsun", "wrong", "chained"),
    ("sun", _S + "sun = img[2,10, 29,37]\nsun", "error", "too-many-parts"),
    ("sun", _S + "sun = img[9:2, 29:37]\nsun", "wrong", None),
    ("red-pixels", _RED_START, "empty", None),
    ("red-pixels", _RED, "correct", None),
    ("red-pixels", _R + "    return len(img[:, :, 0][img[:, :, 0] > limit])", "correct", None),
    ("red-pixels", _R + "    return int((img[:, :, 0] > limit).sum())", "correct", None),
    ("red-pixels", _R + "    n = 0\n    for row in img:\n        for pixel in row:\n            if pixel[0] > limit:\n                n = n + 1\n    return n", "correct", None),
    ("red-pixels", _R + "    red = img[:, :, 0]\n    if red > limit:\n        return len(red)", "error", "if-array"),
    ("red-pixels", _R + "    red = img[:, :, 1]\n    return len(red[red > limit])", "wrong", "green"),
    ("red-pixels", _R + "    red = img[:, :, 2]\n    return len(red[red > limit])", "wrong", "blue"),
    ("red-pixels", _R + "    return len(img[img > limit])", "wrong", "all-channels"),
    ("red-pixels", _R + "    red = img[:, 0]\n    return len(red[red > limit])", "wrong", "column"),
    ("red-pixels", _R + "    red = img[0]\n    return len(red[red > limit])", "wrong", "row"),
    ("red-pixels", _R + "    red = img[:, :, 0]\n    return len(red[limit > red])", "wrong", "less"),
    ("red-pixels", _R + "    red = img[:, :, 0]\n    return len(red[red >= limit])", "wrong", "at-least"),
    ("red-pixels", _R + "    red = img[:, :, 0]\n    return len(red[red > 0.5])", "almost", "fixed-limit"),
    ("red-pixels", _R + "    red = img[:, :, 0]\n    return len(red)", "wrong", "rows-not-pixels"),
    ("red-pixels", _R + "    red = img[:, :, 0]\n    return red[red > limit]", "wrong", "pixels-not-count"),
    ("red-pixels", _R + "    return img[:, :, 0] > limit", "wrong", "mask-itself"),
]
