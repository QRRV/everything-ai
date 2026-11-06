# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-color-spaces: the value and saturation of one pixel, the saturation threshold that leaves
out the pink tablecloth, and a mask for red with the hue at both ends of the circle. And the pictures: the three
numbers of HSV as strips, and the drawing of fruit with what a mask picks. The drawing itself is made by
`draw_fruit`; the test of the lesson checks that the files in data/photos/ are exactly that drawing."""
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value

FRUIT = "data/photos/fruit.png"           # 240 rows, 360 columns, RGB; drawn by the course (draw_fruit)
LABELS = "data/photos/fruit-labels.png"   # the same size: the number of what each pixel is, FRUIT_KINDS
FRUIT_KINDS = ["tablecloth", "apple", "orange", "lemon", "lime"]
APPLE, LIME = 1, 4
LIT, SHADE = (70, 70), (175, 260)         # the middle of an apple in the light, and of one in the shade
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # the stops of the slider

# the fruit: what it is (its number in FRUIT_KINDS), row, column, radius, hue, saturation
_FRUIT = [(1, 70, 70, 34, 0.000, 0.85), (3, 175, 60, 30, 0.150, 0.80), (1, 160, 150, 36, 0.985, 0.85),
          (2, 70, 175, 33, 0.075, 0.90), (4, 75, 285, 28, 0.270, 0.75), (1, 175, 260, 35, 0.000, 0.85),
          (1, 165, 330, 26, 0.990, 0.80)]


def draw_fruit() -> tuple[np.ndarray, np.ndarray]:
    """The drawing of the lesson, made in HSV: fruit on a pink checked tablecloth, under a lamp on the left. Every
    fruit is a ball with light from the upper left, a darker side and a white shine; the lamp gives the right side
    less light. Gives the image as uint8 and the label of every pixel (its number in FRUIT_KINDS)."""
    from matplotlib.colors import hsv_to_rgb
    rows, cols = np.mgrid[0:240, 0:360]
    weave = (np.sin(rows * 0.9) * np.sin(cols * 0.9) + 1) / 2
    stripe = ((cols // 24 + rows // 24) % 2).astype(float)
    hsv = np.stack([(0.99 + 0.03 * stripe + 0.006 * np.sin(rows * 1.7 + cols * 2.3)) % 1.0,   # pink: a red ...
                    0.22 + 0.16 * stripe + 0.08 * weave,                                       # ... with little saturation
                    0.80 + 0.08 * weave - 0.05 * stripe], axis=-1)
    labels = np.zeros((240, 360), dtype=np.uint8)
    for kind, r0, c0, radius, hue, sat in _FRUIT:
        dy, dx = (rows - r0) / radius, (cols - c0) / radius
        inside = dx ** 2 + dy ** 2 < 1
        light = np.clip(-0.45 * dx - 0.55 * dy + 0.70 * np.sqrt(np.clip(1 - dx ** 2 - dy ** 2, 0, 1)), 0, 1)
        shine = np.clip(1 - np.hypot(dx + 0.38, dy + 0.42) / 0.16, 0, 1)
        value = 0.30 + 0.62 * light
        hsv[inside] = np.stack([np.full(inside.sum(), hue), (sat * (1 - shine))[inside],
                                (value + (1 - value) * shine)[inside]], axis=-1)
        labels[inside] = kind
    hsv[..., 2] *= 1.0 - 0.62 * (cols / 359) ** 1.2
    return np.round(hsv_to_rgb(hsv) * 255).astype(np.uint8), labels


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def fruit_hsv() -> np.ndarray:
    """The drawing in HSV, as the worked example makes it."""
    from matplotlib.colors import rgb_to_hsv
    return rgb_to_hsv(read(FRUIT) / 255)


def red_rule(hsv: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """The course's rule for red: hue near 0 or near 1, and saturation above the threshold."""
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    return ((h < 0.05) | (h > 0.95)) & (s > threshold)


def found(mask: np.ndarray) -> dict:
    """What a mask picks on the drawing: the share of the apple pixels, and how many pixels of the tablecloth and of
    the other fruit."""
    labels = read(LABELS)
    return {"apple": float(mask[labels == APPLE].mean()), "cloth": int(mask[labels == 0].sum()),
            "other": int(mask[labels > APPLE].sum())}


def best_threshold() -> float:
    """The lowest stop of the slider at which the rule for red picks no pixel of the tablecloth."""
    hsv = fruit_hsv()
    return min(t for t in THRESHOLDS if found(red_rule(hsv, t))["cloth"] == 0)


def _threshold_outcome(t):
    """What the slider leads to: the share of the apple pixels the rule finds, in percent, to one decimal."""
    return round(100 * found(red_rule(fruit_hsv(), float(t)))["apple"], 1)


def _cloth(t) -> int:
    return found(red_rule(fruit_hsv(), float(t)))["cloth"]


def strips(labels: dict[str, str]) -> str:
    """The three numbers of HSV, each as a strip from 0 to 1: the hue with full saturation and value, then the
    saturation and the value of red."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from everything_ai import plots
    x = np.linspace(0, 1, 361)
    rows = {"hue": np.stack([x, np.ones_like(x), np.ones_like(x)], axis=-1),
            "saturation": np.stack([np.zeros_like(x), x, np.ones_like(x)], axis=-1),
            "value": np.stack([np.zeros_like(x), np.ones_like(x), x], axis=-1)}
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.2), gridspec_kw={"hspace": 1.6})
    for ax, (key, row) in zip(axes, rows.items()):
        ax.imshow(hsv_to_rgb(np.repeat(row[None], 14, axis=0)), aspect="auto", extent=(0, 1, 0, 1))
        ax.set_yticks([])
        ax.set_title(labels[key], fontsize=10)
        if key == "hue":
            ax.set_xticks([0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1], labels["hue_ticks"], fontsize=9)
        else:
            ax.set_xticks([0, 0.5, 1], labels[key + "_ticks"], fontsize=9)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def threshold_frame(t: float, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the drawing with what the rule for red picks at this saturation threshold in its own colors,
    and the rest faded to grey; with what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    img = read(FRUIT)
    mask = red_rule(fruit_hsv(), t)
    shown = img.astype(float)
    shown[~mask] = 0.25 * shown[~mask].mean(axis=1, keepdims=True) + 0.75 * 205
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.imshow(shown.astype(np.uint8), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    html = plots.render(fig, alt=labels["alt"].format(t=t))
    plt.close(fig)
    return html, found(mask)


def grouped(n: int, sep: str) -> str:
    """A whole number with a separator between groups of three digits: 26,038, or 26.038 in Dutch."""
    return f"{n:,}".replace(",", sep)


def _rounded(v) -> str:
    """What value_saturation gave back, as the lesson writes it: numbers with at most six decimals, so that
    0.7500000000000001 shows as 0.75 and two different numbers never look the same; a list as a list."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if x != x:
            return "nan"
        text = f"{x:.6f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    if isinstance(v, tuple):
        return "(" + ", ".join(_rounded(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_rounded(x) for x in v) + "]"
    return repr(v)


def _as_lists(v) -> str:
    """A mask as the lesson writes it: nested lists of True and False."""
    return str(v.tolist()) if isinstance(v, np.ndarray) else repr(v)


def _has_nan(got) -> bool:
    try:
        return any(float(x) != float(x) for x in got)
    except (TypeError, ValueError):
        return False


def _value_above_one(got) -> bool:
    try:
        return float(got[0]) > 1
    except (TypeError, ValueError, IndexError, KeyError):
        return False


# The lesson is done when these are correct. "stronger" is a choice question in the lesson.
REQUIRED = ["stronger", "value-saturation", "threshold", "red-mask"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_VS_HEAD = "import numpy as np\n\n\ndef value_saturation(pixel):\n    rgb = np.array(pixel) / 255\n"
_VS_START = _VS_HEAD + "    ...\n"
_VS = _VS_HEAD + "    v = rgb.max()\n    if v == 0:\n        return 0.0, 0.0\n    s = (v - rgb.min()) / v\n    return v, s"

_RED_HEAD = ("import numpy as np\n\n\ndef red_mask(hsv):\n    hsv = np.array(hsv)\n"
             "    h = hsv[:, :, 0]\n    s = hsv[:, :, 1]\n")
_RED_START = _RED_HEAD + "    ...\n"
_RED = _RED_HEAD + "    return ((h < 0.05) | (h > 0.95)) & (s > 0.5)"
# One row of five pixels in HSV: red, red at the other end of the circle, orange, grey and pink.
ROW = [[[0.02, 0.9, 0.8], [0.98, 0.7, 0.5], [0.08, 0.9, 0.9], [0.0, 0.0, 0.6], [0.01, 0.3, 0.9]]]


EXERCISES = {
    "value-saturation": FunctionBox(
        name="value-saturation",
        function="value_saturation",
        start=_VS_START,
        solution=_VS,
        visible=[([204, 102, 51],)],
        hidden=[([0, 0, 0],), ([128, 128, 128],), ([255, 0, 0],), ([10, 40, 20],), ([255, 200, 200],), ([100, 0, 0],),
                ([255, 255, 255],)],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("not-scaled", matches=lambda got, text: _value_above_one(got),
                    en="`{call}` gives `{got}`: a value above 1, so it is still on the scale from 0 to 255. Use `rgb`, the pixel divided by 255, and not `pixel` itself.",
                    nl="`{call}` geeft `{got}`: een helderheid boven 1, dus nog op de schaal van 0 tot en met 255. Gebruik `rgb`, de pixel gedeeld door 255, en niet `pixel` zelf."),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="`{call}` gives the list `{got}`. Give back the two numbers themselves, separated by a comma: `return v, s`.",
                    nl="`{call}` geeft de lijst `{got}`. Geef de twee getallen zelf terug, gescheiden door een komma: `return v, s`."),
            Mistake("swapped", wrong=_VS_HEAD + "    v = rgb.max()\n    if v == 0:\n        return 0.0, 0.0\n    s = (v - rgb.min()) / v\n    return s, v",
                    en="`{call}` gives `{got}`, and it should give `{want}`: the same two numbers, the other way round. Give back the value first, then the saturation: `return v, s`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dezelfde twee getallen, andersom. Geef eerst de helderheid terug, dan de verzadiging: `return v, s`."),
            Mistake("swapped", wrong=_VS_HEAD + "    v = rgb.max()\n    s = (v - rgb.min()) / v\n    return s, v", en="`{call}` gives `{got}`, and it should give `{want}`: the same two numbers, the other way round. Give back the value first, then the saturation: `return v, s`.", nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dezelfde twee getallen, andersom. Geef eerst de helderheid terug, dan de verzadiging: `return v, s`."),
            Mistake("not-divided", wrong=_VS_HEAD + "    v = rgb.max()\n    return v, v - rgb.min()",
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your second number says how far apart the strongest and the weakest of the three are. Saturation compares that with the strongest: divide it by `v`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je tweede getal zegt hoe ver de sterkste en de zwakste van de drie uit elkaar liggen. De verzadiging vergelijkt dat met de sterkste: deel het door `v`."),
            Mistake("by-min", wrong=_VS_HEAD + "    v = rgb.max()\n    if v == 0:\n        return 0.0, 0.0\n    s = (v - rgb.min()) / rgb.min()\n    return v, s",
                    en="`{call}` gives `{got}`, and it should give `{want}`. You divide by the weakest of the three. Saturation divides by the strongest, `v`, so it stays between 0 and 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je deelt door de zwakste van de drie. De verzadiging deelt door de sterkste, `v`, zodat ze tussen 0 en 1 blijft."),
            Mistake("mean", wrong=_VS_HEAD + "    v = rgb.mean()\n    if v == 0:\n        return 0.0, 0.0\n    s = (v - rgb.min()) / v\n    return v, s",
                    en="`{call}` gives `{got}`, and it should give `{want}`. The value of HSV is not the mean of red, green and blue, but the strongest of the three: `rgb.max()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De helderheid van HSV is niet het gemiddelde van rood, groen en blauw, maar de sterkste van de drie: `rgb.max()`."),
            Mistake("mean", wrong=_VS_HEAD + "    v = rgb.mean()\n    s = (v - rgb.min()) / v\n    return v, s", en="`{call}` gives `{got}`, and it should give `{want}`. The value of HSV is not the mean of red, green and blue, but the strongest of the three: `rgb.max()`.", nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De helderheid van HSV is niet het gemiddelde van rood, groen en blauw, maar de sterkste van de drie: `rgb.max()`."),
            Mistake("black", state="almost", matches=lambda got, text: _has_nan(got),
                    en="`{call}` gives `{got}`: for black, `v` is 0, and 0 divided by 0 gives nan, not a number. A pixel without any light has no color: give back saturation 0 when `v` is 0, with an `if` before you divide.",
                    nl="`{call}` geeft `{got}`: bij zwart is `v` gelijk aan 0, en 0 gedeeld door 0 geeft nan, geen getal. Een pixel zonder licht heeft geen kleur: geef verzadiging 0 terug als `v` 0 is, met een `if` voordat je deelt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the value and the saturation of the pixel, from `rgb`.",
                      "nl": "Vervang de `...` door de helderheid en de verzadiging van de pixel, uit `rgb`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one pixel, and the `def` line of your function has a different number of names between its parentheses. Keep `(pixel)`.",
                          "nl": "De check roept `{call}` aan, met één pixel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(pixel)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The value is the strongest of the three, and the saturation is the strongest minus the weakest, divided by the strongest.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De helderheid is de sterkste van de drie, en de verzadiging is de sterkste min de zwakste, gedeeld door de sterkste."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every pixel, also for black and for grey?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke pixel, ook voor zwart en voor grijs?"},
            "correct": {"en": "Your function is right in all {n} cases. Grey and white get saturation 0, however bright they are, and black gets value 0 and saturation 0: no light, no color.",
                        "nl": "Je functie klopt in alle {n} gevallen. Grijs en wit krijgen verzadiging 0, hoe helder ze ook zijn, en zwart krijgt helderheid 0 en verzadiging 0: geen licht, geen kleur."},
        },
    ),
    "threshold": Value(
        name="threshold",
        start=0.0,
        solution=0.5,          # best_threshold(); the test of the lesson measures it again
        outcome=_threshold_outcome,
        expect_from=lambda: _threshold_outcome(best_threshold()),
        compare=lambda got, want: abs(got - want) < 1e-9,
        prose_numbers=True,
        mistakes=[
            Mistake("too-low", matches=lambda got, t: _cloth(t) > 0,
                    en="With threshold {value}, the mask finds {got}% of the apple pixels, but also pixels of the tablecloth: part of its pink has a saturation above {value}. Move the slider up.",
                    nl="Met drempel {value} vindt het masker {got}% van de appelpixels, maar ook pixels van het tafelkleed: een deel van zijn roze heeft een verzadiging boven {value}. Schuif de slider omhoog."),
            Mistake("higher", state="almost", matches=lambda got, t: float(t) > best_threshold(),
                    en="With threshold {value}, the mask picks not a single pixel of the tablecloth and finds {got}% of the apple pixels. A lower threshold finds more of the apples: can it go lower and still leave out the whole tablecloth?",
                    nl="Met drempel {value} pakt het masker geen enkele pixel van het tafelkleed en vindt het {got}% van de appelpixels. Een lagere drempel vindt meer van de appels: kan hij lager en toch het hele tafelkleed weglaten?"),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at 0, where it started: every red pixel counts, also the pink of the tablecloth. Move it up, and watch the tablecloth.",
                      "nl": "De slider staat nog op 0, waar hij begon: elke rode pixel telt, ook het roze van het tafelkleed. Schuif hem omhoog, en kijk naar het tafelkleed."},
            "wrong": {"en": "With threshold {value}, the mask finds {got}% of the apple pixels. Look for the threshold with the most apple pixels and not a single pixel of the tablecloth.",
                      "nl": "Met drempel {value} vindt het masker {got}% van de appelpixels. Zoek de drempel met de meeste appelpixels en geen enkele pixel van het tafelkleed."},
            "correct": {"en": "Threshold {value} it is: not a single pixel of the tablecloth, and {got}% of the apple pixels. What it misses is the shine of the lamp on the apples, where the color fades into white and the saturation drops below 0.5.",
                        "nl": "Het wordt drempel {value}: geen enkele pixel van het tafelkleed, en {got}% van de appelpixels. Wat het mist, is de glans van de lamp op de appels, waar de kleur overgaat in wit en de verzadiging onder 0,5 zakt."},
        },
    ),
    "red-mask": FunctionBox(
        name="red-mask",
        function="red_mask",
        start=_RED_START,
        solution=_RED,
        visible=[(ROW,)],
        show=_as_lists,
        hidden=[([[[0.5, 0.9, 0.9], [0.97, 0.9, 0.2]], [[0.04, 0.6, 0.1], [0.96, 0.4, 0.8]]],),
                ([[[0.0, 0.0, 0.0]]],),
                ([[[0.99, 0.55, 0.5], [0.06, 0.95, 0.9], [0.3, 0.9, 0.9], [0.02, 0.45, 0.7]]],)],
        first=[_DOTS],
        mistakes=[
            Mistake("and-or", error=lambda e: isinstance(e, ValueError) and "ambiguous" in str(e),
                    en="With `{call}`, your function stops: `and` and `or` work on one `True` or `False`, and here they get a whole mask. For masks, write `&` instead of `and`, and `|` instead of `or`: they go pixel by pixel.",
                    nl="Met `{call}` stopt je functie: `and` en `or` werken op één `True` of `False`, en hier krijgen ze een heel masker. Schrijf voor maskers `&` in plaats van `and`, en `|` in plaats van `or`: die gaan pixel voor pixel."),
            Mistake("parentheses", error=lambda e: isinstance(e, TypeError) and ("bitwise" in str(e) or "unsupported operand" in str(e)),
                    en="With `{call}`, your function stops: Python does `&` and `|` before `<` and `>`, so without parentheses it joins a number and a mask first, as `0.05 | h` in `h < 0.05 | h > 0.95`. Put every comparison in parentheses: `(h < 0.05) | (h > 0.95)`.",
                    nl="Met `{call}` stopt je functie: Python doet `&` en `|` vóór `<` en `>`, dus zonder haakjes verbindt hij eerst een getal met een masker, zoals `0.05 | h` in `h < 0.05 | h > 0.95`. Zet elke vergelijking tussen haakjes: `(h < 0.05) | (h > 0.95)`."),
            Mistake("no-wrap", wrong=_RED_HEAD + "    return (h < 0.05) & (s > 0.5)",
                    en="`{call}` gives `{got}`, and it should give `{want}`. The second pixel has hue 0.98: red too, at the other end of the circle. Red lies at both ends: hue below 0.05 or above 0.95.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De tweede pixel heeft kleurtoon 0,98: ook rood, aan het andere eind van de cirkel. Rood ligt aan beide einden: kleurtoon onder 0,05 of boven 0,95."),
            Mistake("no-saturation", wrong=_RED_HEAD + "    return (h < 0.05) | (h > 0.95)",
                    en="`{call}` gives `{got}`, and it should give `{want}`. The fourth pixel is grey, with hue 0 because it has no color, and the fifth is pink. Leave them out with the saturation: above 0.5.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De vierde pixel is grijs, met kleurtoon 0 omdat hij geen kleur heeft, en de vijfde is roze. Laat ze weg met de verzadiging: boven 0,5."),
            Mistake("order", wrong=_RED_HEAD + "    return (h < 0.05) | (h > 0.95) & (s > 0.5)",
                    en="`{call}` gives `{got}`, and it should give `{want}`. Python does `&` before `|`, as it does × before +: your saturation only counts for the hue above 0.95. Put parentheses around the two conditions on hue.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Python doet `&` vóór `|`, zoals × vóór +: je verzadiging telt alleen voor de kleurtoon boven 0,95. Zet haakjes om de twee voorwaarden op de kleurtoon."),
            Mistake("order", wrong=_RED_HEAD + "    return (s > 0.5) & (h < 0.05) | (h > 0.95)",
                    en="`{call}` gives `{got}`, and it should give `{want}`. Python does `&` before `|`, as it does × before +: your saturation only counts for the hue below 0.05. Put parentheses around the two conditions on hue.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Python doet `&` vóór `|`, zoals × vóór +: je verzadiging telt alleen voor de kleurtoon onder 0,05. Zet haakjes om de twee voorwaarden op de kleurtoon."),
            Mistake("degrees", matches=lambda got, text: re.search(r"\bh\s*[<>]=?\s*(?:[2-9]|[1-9]\d+)(?:\.\d+)?\b", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`. The hue of `rgb_to_hsv` goes from 0 to 1, not from 0 to 360 degrees: red is below 0.05 or above 0.95.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De kleurtoon van `rgb_to_hsv` gaat van 0 tot 1, niet van 0 tot 360 graden: rood is onder 0,05 of boven 0,95."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the mask: `True` for every red pixel, from `h` and `s`.",
                      "nl": "Vervang de `...` door het masker: `True` voor elke rode pixel, uit `h` en `s`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one image in HSV, and the `def` line of your function has a different number of names between its parentheses. Keep `(hsv)`.",
                          "nl": "De check roept `{call}` aan, met één afbeelding in HSV, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(hsv)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Red is hue below 0.05 or above 0.95, and saturation above 0.5: two conditions on hue with `|`, then `&` with the saturation.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Rood is kleurtoon onder 0,05 of boven 0,95, en verzadiging boven 0,5: twee voorwaarden op de kleurtoon met `|`, dan `&` met de verzadiging."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your mask use exactly the conditions of the task: hue below 0.05 or above 0.95, and saturation above 0.5?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je masker precies de voorwaarden van de opdracht: kleurtoon onder 0,05 of boven 0,95, en verzadiging boven 0,5?"},
            "correct": {"en": "Your mask is right in all {n} cases. On the drawing, it finds 99.1% of the apple pixels, in the light and in the shade, and not a single pixel of the tablecloth or the other fruit.",
                        "nl": "Je masker klopt in alle {n} gevallen. Op de tekening vindt het 99,1% van de appelpixels, in het licht en in de schaduw, en geen enkele pixel van het tafelkleed of het andere fruit."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("value-saturation", _VS_START, "empty", None),
    ("value-saturation", _VS, "correct", None),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    s = 0.0 if v == 0 else (v - rgb.min()) / v\n    return v, s", "correct", None),
    ("value-saturation", _VS_HEAD + "    hi = rgb.max()\n    lo = rgb.min()\n    if hi == 0:\n        return 0, 0\n    return hi, 1 - lo / hi", "correct", None),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    s = (v - rgb.min()) / v\n    return v, s", "almost", "black"),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    if v == 0:\n        return 0.0, 0.0\n    return (v - rgb.min()) / v, v", "wrong", "swapped"),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    return v, v - rgb.min()", "wrong", "not-divided"),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    if v == 0:\n        return 0.0, 0.0\n    return v, (v - rgb.min()) / rgb.min()", "wrong", "by-min"),
    ("value-saturation", _VS_HEAD + "    v = rgb.mean()\n    if v == 0:\n        return 0.0, 0.0\n    return v, (v - rgb.min()) / v", "wrong", "mean"),
    ("value-saturation", _VS_HEAD + "    return 0.8, 0.75", "almost", None),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    s = (v - rgb.min()) / v\n    return s, v", "wrong", "swapped"),
    ("value-saturation", _VS_HEAD + "    v = rgb.mean()\n    s = (v - rgb.min()) / v\n    return v, s", "wrong", "mean"),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    if v == 0:\n        return [0.0, 0.0]\n    return [v, (v - rgb.min()) / v]", "wrong", "list"),
    ("value-saturation", _VS_HEAD + "    v = max(pixel)\n    if v == 0:\n        return 0.0, 0.0\n    return v, (v - min(pixel)) / v", "wrong", "not-scaled"),
    ("value-saturation", _VS_HEAD + "    v = rgb.max()\n    s = (v - rgb.min()) / v\n    return f'{v}, {s}'", "wrong", None),
    ("value-saturation", _VS_HEAD + "    v = round(rgb.max(), 3)\n    if v == 0:\n        return 0.0, 0.0\n    return v, round((v - rgb.min()) / v, 3)", "almost", None),
    ("threshold", 0.0, "empty", None),
    ("threshold", 0.5, "correct", None),
    ("threshold", 0.4, "wrong", "too-low"),
    ("threshold", 0.1, "wrong", "too-low"),
    ("threshold", 0.7, "almost", "higher"),
    ("threshold", 0.9, "almost", "higher"),
    ("red-mask", _RED_START, "empty", None),
    ("red-mask", _RED, "correct", None),
    ("red-mask", _RED_HEAD + "    red = (h <= 0.05) | (h >= 0.95)\n    return red & (s > 0.5)", "correct", None),
    ("red-mask", _RED_HEAD + "    return (s > 0.5) & ((h < 0.05) | (h > 0.95))", "correct", None),
    ("red-mask", _RED_HEAD + "    return ((h < 0.05) or (h > 0.95)) and (s > 0.5)", "error", "and-or"),
    ("red-mask", _RED_HEAD + "    return (h < 0.05 | h > 0.95) & (s > 0.5)", "error", "parentheses"),
    ("red-mask", _RED_HEAD + "    return (h < 0.05) & (s > 0.5)", "wrong", "no-wrap"),
    ("red-mask", _RED_HEAD + "    return (h < 0.05) | (h > 0.95)", "wrong", "no-saturation"),
    ("red-mask", _RED_HEAD + "    return (h < 0.05) | (h > 0.95) & (s > 0.5)", "wrong", "order"),
    ("red-mask", _RED_HEAD + "    return (s > 0.5) & (h < 0.05) | (h > 0.95)", "wrong", "order"),
    ("red-mask", _RED_HEAD + "    return ((h < 0.05) | (h > 0.95)) & s > 0.5", "error", "parentheses"),
    ("red-mask", _RED_HEAD + "    return ((h < 18) | (h > 342)) & (s > 0.5)", "wrong", "degrees"),
    ("red-mask", _RED_HEAD + "    return ((h < 0.05) | (h > 0.95)) & (s > 0.4)", "almost", None),
]
