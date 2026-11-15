# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""What the course's server answers at /v1/images/generations (everything_ai.serve): not an image model, but a small
drawing program. It reads the shapes a prompt names, in English or in Dutch ("two red circles and a blue square",
"drie gele driehoeken"), and draws them with the course's shapes (everything_ai.scenes) on a light background, each
image of a request with the shapes in other places. A real image model turns any description into an image; it comes
in the domain on generative models. The course draws, so that the API can be tried without a model of gigabytes.

`read(prompt)` gives the shapes, `said(shapes, lang)` says them in words, `draw(prompt, size, seed)` gives a PNG."""
import hashlib
import io
import math
import re

import numpy as np

from everything_ai import scenes

COLORS = ("red", "blue", "green", "yellow", "purple", "teal")   # the colors of scenes.COLOURS, in that order
_COLOR_WORDS = {"red": 0, "rood": 0, "rode": 0, "blue": 1, "blauw": 1, "blauwe": 1, "green": 2, "groen": 2, "groene": 2,
                "yellow": 3, "geel": 3, "gele": 3, "purple": 4, "paars": 4, "paarse": 4, "teal": 5, "turquoise": 5, "turkoois": 5, "turkooise": 5}
_KIND_WORDS = {"circle": "circle", "circles": "circle", "cirkel": "circle", "cirkels": "circle", "rondje": "circle",
               "rondjes": "circle", "square": "square", "squares": "square", "vierkant": "square", "vierkanten": "square",
               "vierkantje": "square", "vierkantjes": "square", "triangle": "triangle", "triangles": "triangle",
               "driehoek": "triangle", "driehoeken": "triangle", "driehoekje": "triangle", "driehoekjes": "triangle"}
_PLURALS = {"circles", "cirkels", "rondjes", "squares", "vierkanten", "vierkantjes", "triangles", "driehoeken", "driehoekjes"}
_COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "een": 1, "één": 1, "two": 2, "twee": 2, "three": 3, "drie": 3, "four": 4,
                "vier": 4, "five": 5, "vijf": 5, "six": 6, "zes": 6, "seven": 7, "zeven": 7, "eight": 8, "acht": 8, "nine": 9,
                "negen": 9, "ten": 10, "tien": 10}
# words after which a number or a color no longer belongs to the next shape: "a picture of", "on a blue background"
_BREAKS = {"with", "met", "on", "op", "in", "of", "van", "background", "achtergrond", "image", "picture", "photo",
           "drawing", "plaatje", "afbeelding", "foto", "tekening"}
MAX_SHAPES = 8
SIZES = {"256x256": 256, "512x512": 512, "1024x1024": 1024}   # the sizes the server draws, as OpenAI writes them
BACKGROUND = (236, 236, 232)
_NAMES = {"en": {"circle": ("circle", "circles"), "square": ("square", "squares"), "triangle": ("triangle", "triangles")},
          "nl": {"circle": ("cirkel", "cirkels"), "square": ("vierkant", "vierkanten"), "triangle": ("driehoek", "driehoeken")}}
_COLOR_NAMES = {"en": COLORS, "nl": (("rode", "rood"), ("blauwe", "blauw"), ("groene", "groen"), ("gele", "geel"),
                                     ("paarse", "paars"), ("turquoise", "turquoise"))}


def read(prompt: str) -> list[tuple[str, int]]:
    """The shapes the prompt names, as (kind, color) in its order: a number word or a digit before a shape is how many
    (a plural without one, two), a color word its color, until a word such as "with" or "background" comes between. A
    shape without a color gets the next color in turn."""
    shapes, count, color, turn = [], None, None, 0
    for word in re.findall(r"[a-zà-ÿ0-9]+", prompt.lower()):
        if word in _COUNT_WORDS:
            count = _COUNT_WORDS[word]
        elif word.isdigit():   # a long number is more than the server draws, without reading it
            count = min(int(word), MAX_SHAPES + 1) if len(word) <= 3 else MAX_SHAPES + 1
        elif word in _COLOR_WORDS:
            color = _COLOR_WORDS[word]
        elif word in _BREAKS:
            count, color = None, None
        elif word in _KIND_WORDS:
            for _ in range(count if count is not None else 2 if word in _PLURALS else 1):
                if color is None:
                    shapes.append((_KIND_WORDS[word], turn % len(COLORS)))
                    turn += 1
                else:
                    shapes.append((_KIND_WORDS[word], color))
            count, color = None, None
    return shapes


def said(shapes: list[tuple[str, int]], lang: str = "en") -> str:
    """The shapes in words, the same kind and color together in the order they first come: "2 red circles and 1 blue
    square"."""
    groups = {}
    for shape in shapes:
        groups[shape] = groups.get(shape, 0) + 1
    words = []
    for (kind, color), n in groups.items():
        name = _NAMES[lang][kind][n > 1]
        c = _COLOR_NAMES[lang][color]
        if lang == "nl":   # een rode cirkel, twee rode cirkels, een rood vierkant (het-woord, enkelvoud)
            c = c[1] if kind == "square" and n == 1 else c[0]
        words.append(f"{n} {c} {name}")
    last = " and " if lang == "en" else " en "
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + last + words[-1]


def _seed(prompt: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).digest()[:8], "little")


def picture(shapes: list[tuple[str, int]], size: int = 256, seed: int = 0) -> np.ndarray:
    """The shapes on the background, size by size pixels (uint8, with 3 colors), in places that `seed` picks: each
    shape lies apart from the others when it can, and later shapes lie on top. Drawn at twice the size and scaled down,
    so that the edges are smooth."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    big = 2 * size
    radius = big * (0.16 if len(shapes) <= 2 else 0.12 if len(shapes) <= 4 else 0.09)
    image = np.empty((big, big, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    placed = []
    for kind, color in shapes:
        r = radius * rng.uniform(0.8, 1.1)
        for _ in range(200):   # a place apart from the shapes before it, if there is one
            centre = tuple(rng.uniform(r + 4, big - r - 4, 2))
            if all(math.dist(centre, c) > r + q + 6 for c, q in placed):
                break
        placed.append((centre, r))
        top, left = (max(0, int(c - r) - 2) for c in centre)   # the shape is drawn in a window around it only
        w = min(int(2 * r) + 6, big - top, big - left)
        mask, _ = scenes.shape_mask(kind, (centre[0] - top, centre[1] - left), r, rng.uniform(0, 2 * math.pi), w)
        image[top:top + w, left:left + w][mask] = scenes.COLOURS[color]
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BOX))


def draw(prompt: str, size: int = 256, seed: int = 0) -> bytes:
    """A PNG of the shapes the prompt names (see `read`), size by size pixels; another seed puts them in other places.
    It raises ValueError when the prompt names no shape, or more than MAX_SHAPES."""
    from PIL import Image
    shapes = read(prompt)
    if not shapes:
        raise ValueError("nothing")
    if len(shapes) > MAX_SHAPES:
        raise ValueError("many")
    buffer = io.BytesIO()
    Image.fromarray(picture(shapes, size, _seed(prompt, seed))).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
