# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-image-search: search, which finds the rows of a matrix of features most like a vector, by
cosine similarity. It is checked on made-up features, so it works without DINOv2. DINOv2-small (88 MB, from Hugging
Face) and Caltech-101 (137 MB, from CaltechDATA) the lesson downloads once, when the learner asks for them
(everything_ai.hub, everything_ai.caltech), as in cv-pretrained-features: the worked example and the demo need them,
the exercises do not. The slider uses what the course measured with them (REFERENCE)."""
import errno
import functools
import html
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: DINOv2 runs on it

from everything_ai import caltech, chart, dinov2, hub
from everything_ai.lesson import download_card
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit
from everything_ai import yours

DOWNLOAD_LINE = {
    "en": "Works on every computer. DINOv2-small, 88 MB, is downloaded once from Hugging Face, and Caltech-101, 137 MB of photos, from Caltech, when you click their button in the lesson; if you did the previous lesson, they are already there. The exercises work without them.",
    "nl": "Werkt op elke computer. DINOv2-small, 88 MB, wordt één keer gedownload van Hugging Face, en Caltech-101, 137 MB aan foto's, van Caltech, als je in de les op de knop ervoor klikt; heb je de vorige les gedaan, dan staan ze er al. De oefeningen werken ook zonder.",
}
CLASSES = list(caltech.CHECKED)   # the 10 classes of Caltech-101 the lesson uses, in alphabetical order: label 0 to 9
NAMES = {
    "en": {"chandelier": "chandelier", "dragonfly": "dragonfly", "ewer": "ewer", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "rooster", "starfish": "starfish", "sunflower": "sunflower", "water_lilly": "water lily",
           "windsor_chair": "Windsor chair"},
    "nl": {"chandelier": "kroonluchter", "dragonfly": "libel", "ewer": "schenkkan", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "haan", "starfish": "zeester", "sunflower": "zonnebloem", "water_lilly": "waterlelie",
           "windsor_chair": "windsorstoel"},
}
PER = 30                                        # photos per class: 300 in the collection
QUERY = CLASSES.index("lotus") * PER            # the worked example searches with the first lotus: photo 90
K_VALUES = list(range(1, PER))   # the stops of the slider: how many photos most alike, 1 to 29

# What the course measured: everything_ai.dinov2 with its pinned file, the 30 photos of each class
# (everything_ai.caltech.checked), 300 in all. Every photo searches the 299 others by cosine similarity; "features"
# and "pixels" give, for every k of the slider, the percentage of the k photos most alike that are of the same class,
# on DINOv2's features and on the prepared photos themselves (3 x 224 x 224 = 150,528 numbers). "of5" is how many of
# the 5 most alike are of the same class, on average over the 300 photos, and "class5" the same per class. "same"
# and "other" are the average cosine similarity on the features of two photos of the same class and of two of
# different classes. "example" is the worked example: the 5 photos most like photo 90 (after itself), with their
# class and similarity. "yellow" is the percentage of strongly yellow pixels (yellow below) in the sunflower photos
# and the lowest and highest of the other classes, by class. "gray" is the search on the pixels without their color
# (the mean of the three channels): how many of the 5 most alike are of the same class for a sunflower, and over all
# 300 photos. "pairs" is the average cosine similarity on the features of a lotus and a water lily, and of two
# different lotuses. tests/test_cv_image_search.py measures them again when EAI_DINOV2_FILE and
# EAI_CALTECH_DIR name the downloads.
REFERENCE = {
    "features": {1: 95.7, 2: 95.7, 3: 95.7, 4: 95.8, 5: 95.6, 6: 95.6, 7: 95.4, 8: 95.4, 9: 95.3, 10: 95.2, 11: 95.2,
                12: 95.2, 13: 95.0, 14: 94.9, 15: 94.8, 16: 94.6, 17: 94.5, 18: 94.2, 19: 94.1, 20: 94.0, 21: 93.9,
                22: 93.7, 23: 93.4, 24: 93.2, 25: 92.9, 26: 92.6, 27: 92.3, 28: 91.9, 29: 90.9},
    "pixels": {1: 38.7, 2: 35.2, 3: 32.3, 4: 30.2, 5: 28.5, 6: 27.7, 7: 26.0, 8: 25.4, 9: 24.9, 10: 24.3, 11: 23.7,
              12: 23.2, 13: 22.7, 14: 22.1, 15: 21.7, 16: 21.5, 17: 21.3, 18: 20.9, 19: 20.7, 20: 20.4, 21: 20.2,
              22: 20.1, 23: 19.9, 24: 19.6, 25: 19.6, 26: 19.5, 27: 19.2, 28: 19.0, 29: 18.8},
    "of5": {"features": 4.78, "pixels": 1.42},
    "class5": {"features": [5.0, 4.97, 5.0, 3.7, 5.0, 5.0, 5.0, 4.97, 4.17, 5.0],
               "pixels": [1.33, 1.87, 0.93, 1.17, 0.6, 1.97, 1.43, 3.07, 0.77, 1.1]},
    "same": 0.60, "other": 0.05,
    "example": [("lotus", 0.776), ("lotus", 0.776), ("lotus", 0.774), ("water_lilly", 0.749), ("lotus", 0.742)],
    "yellow": {"sunflower": 27, "others": (1, 4)},
    "gray": {"sunflower": 1.3, "of5": 1.25},
    "pairs": {"lotus-lily": 0.50, "lotus-lotus": 0.52},
}
CHANCE = 100 * (PER - 1) / (len(CLASSES) * PER - 1)   # photos picked at random: 29 of the 299 others share the class


def percent(x: float, lang: str) -> str:
    """A percentage with one decimal and the decimal mark of the language: 95.6, or 95,6 in Dutch."""
    text = f"{x:.1f}"
    return text.replace(".", ",") if lang == "nl" else text


def decimal(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def yellow(photo) -> float:
    """The share of strongly yellow pixels in a photo, shrunk to 64 by 64: red above 150, green above 120, blue
    below 90."""
    from PIL import Image
    with Image.open(photo) as img:
        a = np.asarray(img.convert("RGB").resize((64, 64))).astype(int)
    return float(((a[..., 0] > 150) & (a[..., 1] > 120) & (a[..., 2] < 90)).mean())


# --- the picture: the features of a collection, once, and a search ---------------------------------------------------

def search_picture(labels: dict) -> str:
    """Two columns that meet: on the left the collection, through DINOv2 once, into F; on the right the photo you
    search with, through DINOv2, into q (features: hidden; DINOv2 frozen, grey). Below, across both, the cosine
    similarities and the photos most alike. Nothing learns here, so nothing is orange. `labels` has "title",
    "collection", "query", "dino", "once", "F", "q", "sims", "sims_note", "top", "top_note" and "alt"."""
    W, G = 360, 16
    col = (W - G) / 2
    frozen = "fill:var(--ea-ink);fill-opacity:0.06;stroke:var(--ea-ink-3);stroke-width:1.2"
    plain = "fill:none;stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:3 3"
    hidden = "fill:var(--ea-hidden-wash);stroke:var(--ea-hidden);stroke-width:1.2"
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'

    def box(x, y, w, h, style, name, note=""):
        out = f'<rect x="{x + 1}" y="{y}" width="{w - 2}" height="{h}" rx="6" style="{style}"/>'
        if note:
            return out + (f'<text x="{x + 11}" y="{y + 18}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>'
                          f'<text x="{x + 11}" y="{y + 34}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(note)}</text>')
        return out + f'<text x="{x + 11}" y="{y + h / 2 + 4.5}" style="font-size:12.5px;fill:var(--ea-ink)">{html.escape(name)}</text>'

    def down(x, y):
        return f'<path d="M{x} {y + 3} V{y + 15} M{x - 5} {y + 10} L{x} {y + 15} L{x + 5} {y + 10}" {arrow}/>'

    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = 28
    for x, first, note in ((0, labels["collection"], labels["once"]), (col + G, labels["query"], "")):
        parts.append(box(x, y, col, 32, plain, first))
        parts.append(down(x + col / 2, y + 32))
        parts.append(box(x, y + 50, col, 44, frozen, labels["dino"], note or " "))
        parts.append(down(x + col / 2, y + 94))
        parts.append(box(x, y + 112, col, 32, hidden, labels["F"] if x == 0 else labels["q"]))
    y += 144
    # the two columns meet in the full-width box below
    parts.append(f'<path d="M{col / 2} {y + 3} V{y + 9} H{W / 2 - 8} M{col + G + col / 2} {y + 3} V{y + 9} H{W / 2 + 8} '
                 f'M{W / 2} {y + 9} V{y + 19} M{W / 2 - 5} {y + 14} L{W / 2} {y + 19} L{W / 2 + 5} {y + 14}" {arrow}/>')
    y += 22
    parts.append(box(0, y, W, 44, plain, labels["sims"], labels["sims_note"]))
    parts.append(down(W / 2, y + 44))
    y += 62
    parts.append(box(0, y, W, 44, plain, labels["top"], labels["top_note"]))
    y += 48
    alt = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {y}" role="img" aria-label="{html.escape(alt)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg></figure>")


# --- the slider: how many of the k photos most alike are of the same class --------------------------------------------

def curve(k: int, labels: dict, lang: str) -> str:
    """The percentage of the k photos most alike that are of the same class, against k: on DINOv2's features (solid)
    and on the pixels (dashed), with photos picked at random at about 10. The stop k is marked on both lines.
    `labels` has "title", "x", "y", "features", "pixels", "chance" and "alt"."""
    xs = K_VALUES
    panel = chart.Panel((0.8, 36), (0, 100), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                        xticks=[1, 2, 5, 10, 29], yticks=[0, 25, 50, 75, 100], xlog=True)
    panel.line([0.8, 36], [CHANCE, CHANCE], role="ink-3", width=1.2, dashed=True)
    panel.line(xs, [REFERENCE["features"][x] for x in xs], role="pred")
    panel.line(xs, [REFERENCE["pixels"][x] for x in xs], role="pred", dashed=True)
    panel.points([k], [REFERENCE["features"][k]], role="pred", r=5)
    panel.points([k], [REFERENCE["pixels"][k]], role="pred", r=5, hollow=True)
    panel.text(1.3, 83, labels["features"], role="ink", anchor="start")
    panel.text(1.3, 50, labels["pixels"], role="ink", anchor="start")
    panel.text(1.3, 19, labels["chance"], role="ink-3", anchor="start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- DINOv2 and Caltech-101 in the lesson -----------------------------------------------------------------------------

def photos(name: str) -> list:
    """The 30 photos of a class the lesson uses, without people: everything_ai.caltech.checked."""
    return caltech.checked(name, PER)


def ready() -> bool:
    return dinov2.downloaded() and caltech.unpacked()


@functools.lru_cache(maxsize=1)
def _dino():
    return dinov2.load()


def dino_or_stop(button, texts: dict):
    """DINOv2-small for the lesson's cells, with Caltech-101 unpacked, as in cv-pretrained-features. When either is not
    there yet, the cell stops with `texts["ask"]` and the button; a click downloads what is missing and unpacks the
    photos, with the progress in the cell. While it downloads, the lesson's other cells wait. `texts` has "ask",
    "busy" ({name}, {done} and {total} in MB), "unpacking" ({done} photos), "no_net", "not_pinned", "no_room" and
    "failed" ({why}): why it did not work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            for name, pin in (("DINOv2-small", dinov2.FILE), ("Caltech-101", caltech.ZIP)):
                if hub.pinned(*pin) is not None:
                    continue
                total = round(hub.PINNED[pin]["size"] / 1e6)
                last = [-1]

                def progress(done, size, name=name, total=total, last=last):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(name=name, done=mb, total=total)))
                hub.download(*pin, progress=progress)
            caltech.unpack(progress=lambda done: mo.output.replace(mo.md(texts["unpacking"].format(done=done))))
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    model = _dino()
    mo.output.replace(mo.md(texts["here"]))
    return model


@functools.lru_cache(maxsize=1)
def collection(model):
    """The collection of the worked example and the demo: the features `model` (DINOv2) gives the 30 photos of every
    class, computed once, as a NumPy array F of shape (300, 384), a row per photo in the order of CLASSES; the label
    of every photo, the place of its class in CLASSES; and the files of the photos. The 30 photos of a class go
    through the model together, which is faster than one by one and gives the same features."""
    files = [p for name in CLASSES for p in photos(name)]
    with torch.no_grad():
        F = torch.cat([model(torch.cat([dinov2.prepare(p) for p in photos(name)])) for name in CLASSES]).numpy()
    return F, np.repeat(np.arange(len(CLASSES)), PER), files


def results(files: list, labels, sims, positions, lang: str):
    """The photos at `positions` in a row, the first being the photo searched with, each with its class and its
    cosine similarity, as the worked example shows them: as HTML marimo shows."""
    import marimo as mo
    import matplotlib.pyplot as plt
    from PIL import Image
    from everything_ai import plots
    positions = [int(j) for j in positions]
    fig, axes = plt.subplots(1, len(positions), figsize=(6.6, 1.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.01, wspace=0.08)
    shown = []
    for ax, j in zip(np.atleast_1d(axes), positions):
        with Image.open(files[j]) as img:
            img = img.convert("RGB")
            side = min(img.size)
            left, top = (img.size[0] - side) // 2, (img.size[1] - side) // 2
            ax.imshow(np.asarray(img.crop((left, top, left + side, top + side)).resize((160, 160))))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        name = NAMES[lang][CLASSES[int(labels[j])]]
        ax.set_title(f"{name}\n{decimal(float(sims[j]), lang, 3)}", fontsize=9)
        shown.append(f"{name} {decimal(float(sims[j]), lang, 3)}")
    first = {"en": "The photo searched with and the photos most like it, with their class and cosine similarity: ",
             "nl": "De foto waarmee gezocht is en de foto's die er het meest op lijken, met hun klasse en cosine similarity: "}
    out = plots.render(fig, alt=first[lang] + ", ".join(shown) + ".")
    plt.close(fig)
    return mo.Html(out)


# The lesson is done when these are correct. "sims" and "pixels" are the choice questions, "look-alike" the open one.
REQUIRED = ["sims", "search", "pixels", "look-alike"]

# --- Exercise 2: search -----------------------------------------------------------------------------------------------

_HEAD = "import numpy as np\n\n\ndef search(F, q, k):\n"
_SIMS = "(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))"
_ORDER = 'np.argsort(-sims, kind="stable")[:k]'


def _search(sims: str = _SIMS, order: str = _ORDER, comments: bool = False) -> str:
    first = "    # the cosine similarity of q with every row of F\n" if comments else ""
    second = "    # the positions of the k largest, the largest first\n" if comments else ""
    return _HEAD + first + f"    sims = {sims}\n" + second + f"    return {order}\n"


_START = _search(sims="...", order="...", comments=True)
_SOLUTION = _search()


class _Data:
    """Made-up features as a test case: written as their name in the check's messages; the function gets a copy of the
    array itself."""

    def __init__(self, shown: str, value: np.ndarray):
        self.shown, self.value = shown, value

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return self.value.copy()


def _made_up(n: int, d: int, k: int, seed: int):
    """n rows of d features, of very different lengths, and a vector q: the dot product, the distance and the cosine
    similarity can put the rows in different orders."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, d)) * rng.uniform(0.2, 5, size=(n, 1))
    q = rng.normal(size=d) * 2
    return (_Data("F", F), _Data("q", q), k)


def _summary(got):
    """What the check compares: whole numbers, such as positions, as a list; other numbers as a list marked
    "numbers"; anything else as it is."""
    try:
        a = np.asarray(got)
    except Exception:
        return got
    if a.dtype.kind in "iu" and a.ndim <= 1:
        return [int(v) for v in a.ravel()]
    if a.dtype.kind == "f" and a.ndim <= 1:
        return ["numbers", [float(f"{v:.3g}") for v in a.ravel()]]
    return got


def _shown(v):
    if isinstance(v, list) and len(v) == 2 and v[0] == "numbers":
        return {"en": f"the numbers {v[1]}", "nl": f"de getallen {v[1]}"}
    if isinstance(v, list):
        return {"en": f"the positions {v}", "nl": f"de posities {v}"}
    return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_CALL = "`{call}` gives {got}, and it should give {want}"
_CALL_NL = "`{call}` geeft {got}, en dat moet {want} zijn"
_CASES = [_made_up(6, 3, 3, seed=0), _made_up(10, 5, 4, seed=1), _made_up(8, 384, 2, seed=2), _made_up(5, 4, 5, seed=3)]

EXERCISES = {
    "search": FunctionBox(
        name="search",
        picture=yours.steps(call=0),
        function="search",
        start=_START,
        solution=_SOLUTION,
        visible=_CASES[:1],
        hidden=_CASES[1:],
        collect=_summary,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("axis-0", syntax=r"np\.linalg\.norm\(\s*F\s*,\s*axis\s*=\s*0",
                    en="Your code stops with an error at `{line}`: `axis=0` gives a norm per column, one per feature, and you need one per row, one per photo: `np.linalg.norm(F, axis=1)`.",
                    nl="Je code stopt met een fout bij `{line}`: `axis=0` geeft een norm per kolom, één per feature, en je hebt er één per rij nodig, één per foto: `np.linalg.norm(F, axis=1)`."),
            Mistake("q-first", error=lambda e: isinstance(e, ValueError) and "matmul" in str(e),
                    en="With `{call}`, your function stops at `@`: the sizes do not fit. Put the matrix first: `F @ q` is the dot product of `q` with every row of `F`, one number per row.",
                    nl="Met `{call}` stopt je functie bij `@`: de afmetingen passen niet. Zet de matrix vooraan: `F @ q` is het inproduct van `q` met elke rij van `F`, één getal per rij."),
            Mistake("one-norm", matches=lambda got, text: re.search(r"np\.linalg\.norm\(\s*F\s*\)", text) is not None,
                    en=_CALL + ". `np.linalg.norm(F)` is one number, the norm of the whole matrix, so dividing by it leaves the order of the dot product: a long row wins even when it points another way. Divide by the norm of every row, `np.linalg.norm(F, axis=1)`, and by the norm of `q`.",
                    nl=_CALL_NL + ". `np.linalg.norm(F)` is één getal, de norm van de hele matrix, dus delen erdoor laat de volgorde van het inproduct staan: een lange rij wint, ook als hij een andere kant op wijst. Deel door de norm van elke rij, `np.linalg.norm(F, axis=1)`, en door de norm van `q`."),
            Mistake("rows", matches=lambda got, text: getattr(got, "ndim", 0) == 2,
                    en="`{call}` gives back rows of `F` themselves, and the check wants their positions: return what `np.argsort` gives, without `F[...]` around it.",
                    nl="`{call}` geeft rijen van `F` zelf terug, en de check wil hun posities: geef terug wat `np.argsort` geeft, zonder `F[...]` eromheen."),
            Mistake("dot", wrong=_search(sims="F @ q"),
                    en=_CALL + ". Your search puts the rows in the order of their dot product with `q`, so a long row wins even when it points another way. Divide by the norm of every row and of `q`: `np.linalg.norm(F, axis=1) * np.linalg.norm(q)`.",
                    nl=_CALL_NL + ". Je zoekfunctie zet de rijen op volgorde van hun inproduct met `q`, dus een lange rij wint, ook als hij een andere kant op wijst. Deel door de norm van elke rij en van `q`: `np.linalg.norm(F, axis=1) * np.linalg.norm(q)`."),
            Mistake("ascending", wrong=_search(order='np.argsort(sims, kind="stable")[:k]'),
                    en=_CALL + ": you give the rows least like `q`. `np.argsort` sorts from small to large, and a large cosine similarity means alike. Sort the similarities with a minus sign in front: `np.argsort(-sims, kind=\"stable\")`.",
                    nl=_CALL_NL + ": je geeft de rijen die het minst op `q` lijken. `np.argsort` sorteert van klein naar groot, en een grote cosine similarity betekent dat ze op elkaar lijken. Sorteer de similarities met een minteken ervoor: `np.argsort(-sims, kind=\"stable\")`."),
            Mistake("reversed", wrong=_search(order='np.argsort(sims, kind="stable")[-k:]'),
                    en=_CALL + ": the right rows, but the least alike first. The check wants the one most like `q` first: `np.argsort(-sims, kind=\"stable\")[:k]`.",
                    nl=_CALL_NL + ": de goede rijen, maar de minst gelijkende eerst. De check wil de rij die het meest op `q` lijkt vooraan: `np.argsort(-sims, kind=\"stable\")[:k]`."),
            Mistake("distance", wrong=_search(sims="np.linalg.norm(F - q, axis=1)", order='np.argsort(sims, kind="stable")[:k]'),
                    en=_CALL + ". Your search puts the rows in the order of their distance to `q`, as in k-NN. Here the rows have very different lengths, and the search should look only at the direction: the cosine similarity, `(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`.",
                    nl=_CALL_NL + ". Je zoekfunctie zet de rijen op volgorde van hun afstand tot `q`, zoals bij k-NN. Hier hebben de rijen heel verschillende lengtes, en de zoekfunctie moet alleen naar de richting kijken: de cosine similarity, `(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`."),
            Mistake("six", wrong=_search(order='np.argsort(-sims, kind="stable")[:6]'),
                    en=_CALL + ": you keep 6, as in the worked example. Keep `k`: `[:k]`.",
                    nl=_CALL_NL + ": je houdt er 6, zoals in het uitgewerkte voorbeeld. Houd er `k`: `[:k]`."),
            Mistake("all", wrong=_search(order='np.argsort(-sims, kind="stable")'),
                    en=_CALL + ": every row, in the right order. Keep only the first `k`: `[:k]`.",
                    nl=_CALL_NL + ": elke rij, in de goede volgorde. Houd alleen de eerste `k`: `[:k]`."),
            Mistake("skip-first", wrong=_search(order='np.argsort(-sims, kind="stable")[1:k + 1]'),
                    en=_CALL + ": you skip the row most like `q`. In the worked example the first was the photo itself, because it was in the collection; here `q` is not one of the rows of `F`. Keep the first: `[:k]`.",
                    nl=_CALL_NL + ": je slaat de rij over die het meest op `q` lijkt. In het uitgewerkte voorbeeld was de eerste de foto zelf, omdat die in de verzameling zat; hier is `q` geen van de rijen van `F`. Houd de eerste: `[:k]`."),
            Mistake("one-more", wrong=_search(order='np.argsort(-sims, kind="stable")[:k + 1]'),
                    en=_CALL + ": one row too many. `[:k]` keeps the positions 0 to k − 1: exactly `k`.",
                    nl=_CALL_NL + ": één rij te veel. `[:k]` houdt de posities 0 tot en met k − 1: precies `k`."),
            Mistake("similarities", matches=lambda got, text: isinstance(got, list) and len(got) == 2 and got[0] == "numbers",
                    en="`{call}` gives {got}: similarities, and the check wants the positions of the rows. `np.argsort` turns the similarities into positions: `np.argsort(-sims, kind=\"stable\")[:k]`.",
                    nl="`{call}` geeft {got}: similarities, en de check wil de posities van de rijen. `np.argsort` maakt van de similarities posities: `np.argsort(-sims, kind=\"stable\")[:k]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: `sims` is the cosine similarity of `q` with every row of `F`, and the function gives back the positions of the `k` largest, the largest first.",
                      "nl": "Vervang de twee `...`: `sims` is de cosine similarity van `q` met elke rij van `F`, en de functie geeft de posities van de `k` grootste terug, de grootste eerst."},
            "arguments": {"en": "The check calls `{call}`, with the features of the collection, the features you search with and how many rows to give back, and the `def` line of your function has another number of names between its parentheses. Keep `(F, q, k)`.",
                          "nl": "De check roept `{call}` aan, met de features van de verzameling, de features waarmee je zoekt en hoeveel rijen je teruggeeft, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(F, q, k)`."},
            "wrong": {"en": _CALL + ": the positions of the 3 rows of `F` with the largest cosine similarity to `q`, the largest first. Steps 3 and 4 of the recipe?",
                      "nl": _CALL_NL + ": de posities van de 3 rijen van `F` met de grootste cosine similarity met `q`, de grootste eerst. Stap 3 en 4 van het recept?"},
            "almost": {"en": "Right for the example. But " + _CALL + ". Does your function work for any number of rows, features and `k`?",
                       "nl": "Goed voor het voorbeeld. Maar " + _CALL_NL + ". Werkt je functie voor elk aantal rijen, features en `k`?"},
            "correct": {"en": "Your function is right in all {n} cases: the cosine similarity of `q` with every row at once, and the positions of the `k` largest. The demo below lets it search the 300 photos.",
                        "nl": "Je functie klopt in alle {n} gevallen: de cosine similarity van `q` met elke rij tegelijk, en de posities van de `k` grootste. De demo hieronder laat hem de 300 foto's doorzoeken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("search", _START, "empty", None),
    ("search", _SOLUTION, "correct", None),
    ("search", _search(comments=True), "correct", None),
    ("search", _search(sims="np.dot(F, q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))",
                       order="np.argsort(sims)[::-1][:k]"), "correct", None),
    ("search", _search(sims="(F @ q) / np.linalg.norm(F, axis=1)"), "correct", None),
    ("search", _search(sims="(F @ q) / (np.linalg.norm(F, axis=0) * np.linalg.norm(q))"), "error", "axis-0"),
    ("search", _search(sims="F @ q"), "wrong", "dot"),
    ("search", _search(sims="(F @ q) / (np.linalg.norm(F) * np.linalg.norm(q))"), "wrong", "one-norm"),
    ("search", _search(sims="(F @ q) / np.linalg.norm(F)"), "wrong", "one-norm"),
    ("search", _search(sims="(q @ F) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))"), "error", "q-first"),
    ("search", _search(order='F[np.argsort(-sims, kind="stable")[:k]]'), "wrong", "rows"),
    ("search", _search(order="np.argsort(-sims)[:6]"), "wrong", "six"),
    ("search", _search(order='np.argsort(sims, kind="stable")[:k]'), "wrong", "ascending"),
    ("search", _search(order="np.argsort(sims)[-k:]"), "wrong", "reversed"),
    ("search", _search(sims="np.linalg.norm(F - q, axis=1)", order="np.argsort(sims)[:k]"), "wrong", "distance"),
    ("search", _search(order='np.argsort(-sims, kind="stable")'), "wrong", "all"),
    ("search", _search(order="np.argsort(-sims)[1:k + 1]"), "wrong", "skip-first"),
    ("search", _search(order="np.argsort(-sims)[:k + 1]"), "wrong", "one-more"),
    ("search", _search(order="sims"), "wrong", "similarities"),
    ("search", _search(order="..."), "wrong", "dots-left"),
]


# --- the demo: the learner's search on DINOv2's features of the 300 photos --------------------------------------------

def course_search():
    return run_code(_SOLUTION).names["search"]


def demo(model, text) -> dict:
    """The learner's search once it is right, else the course's, on the features `model` (DINOv2) gives the 300
    photos: every photo searches for the 6 rows most like it, and the 5 after itself count. For every class, how many
    of those 5 are of the same class, on average, and whose function it was. If the learner's function fails here
    after all, the course's runs instead. The cell passes the model it got from dino_or_stop, so that it runs only once
    the downloads are there."""
    ran = run_code(text) if text else None
    fn = ran.names.get("search") if ran is not None and ran.error is None else None
    yours = callable(fn)
    fn = fn if yours else course_search()
    F, labels, _ = collection(model)

    def runs(f):
        same = np.zeros(len(CLASSES))
        for i in range(len(F)):
            found = [int(j) for j in f(F.copy(), F[i].copy(), 6) if int(j) != i][:5]
            same[labels[i]] += sum(1 for j in found if labels[j] == labels[i])
        return [float(s / PER) for s in same]
    try:
        return {"class5": with_time_limit(lambda: runs(fn), 60), "yours": yours}
    except Exception:
        return {"class5": with_time_limit(lambda: runs(course_search()), 60), "yours": False}
