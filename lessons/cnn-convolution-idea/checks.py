# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cnn-convolution-idea: convolve, a small filter of k by k weights that slides over an image and
gives a feature map. And the picture: a handwritten digit and its feature maps for a vertical and a horizontal edge
filter."""
import functools
import html
import re

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
VERTICAL = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])     # dark on the left, light on the right
HORIZONTAL = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])   # dark above, light below


@functools.lru_cache(maxsize=1)
def _images():
    return np.load(paths.ROOT / DATA)["train_images"] / 255


def digit(i: int):
    """Training image i of QMNIST as an array of 28 by 28, from 0 for paper to 1 for ink."""
    return _images()[i].reshape(28, 28)


def convolve(img, K):
    """The course's own convolution: the filter K, of k by k, slides over img without going past its edges."""
    img, K = np.asarray(img, dtype=float), np.asarray(K, dtype=float)
    k = len(K)
    out = np.zeros((len(img) - k + 1, len(img[0]) - k + 1))
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[r, c] = (img[r:r + k, c:c + k] * K).sum()
    return out


# --- the picture: an image and its feature maps as squares, in shades of their roles, so they follow the theme ------

_SHADES = [0.3, 0.55, 0.8, 1.0]


def _grid(values, x0: float, y0: float, size: float, signed: bool, top: float) -> str:
    """One array as SVG squares. Unsigned (the image, 0 to 1): the data role. Signed (a feature map, inside the
    model): magenta where the filter's pattern is there, slate where its opposite is, the page for about 0."""
    values = np.asarray(values, dtype=float)
    rows, cols = values.shape
    parts = {}
    for r in range(rows):
        for c in range(cols):
            v = values[r, c] / top
            if abs(v) < 0.1:
                continue
            key = (min(3, int(abs(v) * 4)), v > 0)
            parts.setdefault(key, []).append(f"M{x0 + c * size:g} {y0 + r * size:g}h{size:g}v{size:g}h{-size:g}z")
    out = [f'<rect x="{x0 - 0.5:g}" y="{y0 - 0.5:g}" width="{cols * size + 1:g}" height="{rows * size + 1:g}" '
           f'style="fill:none;stroke:var(--ea-line);stroke-width:1"/>']
    for (level, positive), d in sorted(parts.items()):
        colour = ("var(--ea-hidden)" if positive else "var(--ea-ink-3)") if signed else "var(--ea-data)"
        out.append(f'<path d="{"".join(d)}" style="fill:{colour};fill-opacity:{_SHADES[level]}"/>')
    return "".join(out)


def picture(labels: dict[str, str]) -> str:
    """Image 0 of QMNIST and its feature maps for the vertical and the horizontal edge filter. Each is a square of its
    own with its caption in HTML, so the captions keep their size on a phone, where the squares wrap to a new row. A map
    is one cell smaller than the image on every side, as its number (r, c) belongs to pixel (r + 1, c + 1)."""
    img = digit(0)
    maps = [convolve(img, VERTICAL), convolve(img, HORIZONTAL)]
    top = max(float(np.max(np.abs(m))) for m in maps)
    size = 6
    side = 28 * size + 2
    panels = []
    for values, signed, caption in [(img, False, labels["image"]), (maps[0], True, labels["vertical"]), (maps[1], True, labels["horizontal"])]:
        off = 1 + (size if signed else 0)
        panels.append(f'<div style="flex:0 0 clamp(8rem, 30%, {side}px)">'
                      f'<svg viewBox="0 0 {side} {side}" aria-hidden="true" style="display:block;width:100%;height:auto">'
                      f'{_grid(values, off, off, size, signed, top if signed else 1.0)}</svg>'
                      f'<div style="margin-top:0.3rem;text-align:center;font-size:13px;color:var(--ea-ink-2)">{html.escape(caption)}</div></div>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:{3 * side + 64}px;font-family:var(--ea-font-ui)">'
            f'<div style="font-size:15px;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:0.75rem 1.5rem">{"".join(panels)}</div></figure>')


# The lesson is done when these are correct. "weights" and "which-filter" are choice questions in the lesson.
REQUIRED = ["weights", "convolve", "which-filter"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

def _one_line(v):
    """An array as a list on one line, whole numbers without .0, for the check's messages."""
    try:
        a = np.round(np.asarray(v, dtype=float), 4)
    except (TypeError, ValueError):
        return repr(v)
    return repr(a.astype(int).tolist() if np.all(a == np.round(a)) else a.tolist())


# --- exercise 2: convolve -----------------------------------------------------------------------------------------------

_C_HEAD = "import numpy as np\n\n\ndef convolve(img, K):\n    img = np.array(img)\n    K = np.array(K)\n    k = len(K)\n"
_C_SIZE = "    rows = len(img) - k + 1\n    cols = len(img[0]) - k + 1\n"
_C_MID = "    out = np.zeros((rows, cols))\n    for r in range(rows):\n        for c in range(cols):\n"
_C_STEP = "            out[r, c] = (img[r:r + k, c:c + k] * K).sum()\n"
_C_END = "    return out\n"
_C_START = _C_HEAD + "    rows = ...\n    cols = ...\n" + _C_MID + "            out[r, c] = ...\n" + _C_END


def _c(size: str = _C_SIZE, step: str = _C_STEP) -> str:
    return _C_HEAD + size + _C_MID + step + _C_END


_C = _c()

EXERCISES = {
    "convolve": FunctionBox(
        name="convolve",
        function="convolve",
        start=_C_START,
        solution=_C,
        visible=[([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], [[1, 0], [0, -1]])],
        hidden=[([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]], VERTICAL.tolist()),
                ([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [0, 1, 0, 1, 0, 1], [2, 2, 2, 2, 2, 2]], HORIZONTAL.tolist()),
                ([[3, 1], [4, 1]], [[2]]),
                ([[1, 2], [3, 4]], [[1, 1], [1, 1]])],
        show=lambda v: _one_line(v),
        first=[_DOTS],
        mistakes=[
            Mistake("no-sum", error=lambda e: isinstance(e, ValueError) and "sequence" in str(e),
                    en="Python stops at `out[r, c] = ...`: the right side gives more than one number, and `out[r, c]` holds just one. Add up all the products of the piece and the filter into one number, with `.sum()` and no axis.",
                    nl="Python stopt bij `out[r, c] = ...`: de rechterkant geeft meer dan één getal, en `out[r, c]` bevat er maar één. Tel alle producten van het stuk en het filter op tot één getal, met `.sum()` zonder axis."),
            # a piece of the wrong shape: img[r:r + k][c:c + k], only rows, a fixed 3 instead of k, or too many places
            Mistake("piece-shape", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="With `{call}`, Python stops when it multiplies: the piece of the image and the filter do not have the same shape. The piece should be `k` rows by `k` columns from row `r` and column `c`, both slices in one pair of square brackets, and fit inside the image: `rows` places down and `cols` across.",
                    nl="Met `{call}` stopt Python bij het vermenigvuldigen: het stuk afbeelding en het filter hebben niet dezelfde shape. Het stuk moet `k` rijen en `k` kolommen zijn vanaf rij `r` en kolom `c`, met beide slices tussen één paar vierkante haken, en binnen de afbeelding passen: `rows` plekken omlaag en `cols` opzij."),
            Mistake("one-short", wrong=_c(size="    rows = len(img) - k\n    cols = len(img[0]) - k\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: one row and one column too few. A filter of k fits at `len(img) - k + 1` places: the last start is at `len(img) - k`, and the first at 0.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: één rij en één kolom te weinig. Een filter van k past op `len(img) - k + 1` plekken: het laatste begin is `len(img) - k`, en het eerste 0."),
            Mistake("matmul", wrong=_c(step="            out[r, c] = (img[r:r + k, c:c + k] @ K).sum()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `@` is a matrix product, which mixes rows with columns. A filter multiplies each number with the weight in the same place: `*`, then `.sum()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `@` is een matrixvermenigvuldiging, die rijen met kolommen mengt. Een filter vermenigvuldigt elk getal met de weight op dezelfde plek: `*`, dan `.sum()`."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the three `...`: how many places the filter fits down and across, and for each place the sum of the image piece times the filter.",
                      "nl": "Vervang de drie `...`: op hoeveel plekken het filter past, omlaag en opzij, en voor elke plek de som van het stuk afbeelding keer het filter."},
            "arguments": {"en": "The check calls `{call}`, with an image and a filter, and the `def` line of your function has another number of names between its parentheses. Keep `(img, K)`.",
                          "nl": "De check roept `{call}` aan, met een afbeelding en een filter, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, K)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every place get the sum of the products of the image piece of k by k and the filter, number by number?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Krijgt elke plek de som van de producten van het stuk afbeelding van k bij k en het filter, getal voor getal?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `k` everywhere, and the length of the rows for the columns?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie overal `k`, en de lengte van de rijen voor de kolommen?"},
            "correct": {"en": "Your function is right in all {n} cases, with filters of 1 by 1, 2 by 2 and 3 by 3, and with an image that is not square. One `convolve` works for a filter of any size.",
                        "nl": "Je functie klopt in alle {n} gevallen, met filters van 1 bij 1, 2 bij 2 en 3 bij 3, en met een afbeelding die niet vierkant is. Eén `convolve` werkt voor een filter van elke grootte."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("convolve", _C_START, "empty", None),
    ("convolve", _c(step="            piece = img[r:r + k, c:c + k]\n            out[r, c] = np.sum(piece * K)\n"), "correct", None),
    ("convolve", _c(step="            out[r, c] = img[r:r + k, c:c + k] * K\n"), "error", "no-sum"),
    ("convolve", _c(step="            out[r, c] = (img[r:r + k, c:c + k] * K).sum(axis=0)\n"), "error", "no-sum"),
    ("convolve", _c(step="            out[r, c] = (img[r:r + k][c:c + k] * K).sum()\n"), "error", "piece-shape"),
    ("convolve", _c(step="            out[r, c] = (img[r:r + 3, c:c + 3] * K).sum()\n"), "error", "piece-shape"),
    ("convolve", _c(size="    rows = len(img) - k\n    cols = len(img[0]) - k\n"), "wrong", "one-short"),
    ("convolve", _c(step="            out[r, c] = (img[r:r + k, c:c + k] @ K).sum()\n"), "wrong", "matmul"),
    ("convolve", _c(size="    rows = len(img) - k + 1\n    cols = len(img) - k + 1\n"), "almost", None),
]
