# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-inside-cnn: filter_image, one filter of a CNN's first layer as a small color image with
gray for a weight of 0, and feature_maps, what each stage of ResNet-18 gives for a photo; each saved in the learner's
project (work/projects/cnn/filters.py and maps.py) when right. Both are checked on made-up filters and on a
ResNet-18 with random weights, so they work without the download. ResNet-18 itself, with torchvision's weights trained
on ImageNet, 47 MB, the lesson downloads once when the learner asks for it, as cv-try-classifier does: the worked
example, the slider over the 64 filters and the demo need it. The numbers the text names are in REFERENCE, measured
with the course's code; tests/test_cv_inside_cnn.py measures them again when EAI_RESNET18_FILE names the file."""
import functools
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import models, paths, projects
from everything_ai.check import Mistake, ScenarioBox, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_inside_cnn",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TRY = _lesson_checks("cv-try-classifier")       # the download of ResNet-18, and its button's texts
_PREP = _lesson_checks("cv-prepare-photo")       # the five steps that get a photo ready for it
resnet_or_stop = _TRY.resnet_or_stop

DOWNLOADS = {"ResNet-18": models.RESNET18}   # what the lesson downloads, for the line under its header
PHOTOS = {"cat": paths.ROOT / "data" / "photos" / "chelsea.png", "coffee": paths.ROOT / "data" / "photos" / "coffee.png",
          "fruit": paths.ROOT / "data" / "photos" / "fruit.png"}
STAGES = ("conv1", "layer1", "layer2", "layer3", "layer4")   # what feature_maps gives, in its order
CHANNELS = (64, 64, 128, 256, 512)   # maps per stage
SIDES = (112, 56, 28, 14, 7)         # rows and columns of each map, for a photo of 224 by 224
STRIDES = (2, 4, 8, 16, 32)          # pixels of the photo between two neighboring numbers of a map
VIEWS = (7, 43, 99, 211, 435)        # the receptive field of one number of each stage, in pixels; 11 after the pooling
POOLED = 11
SHOWN = 8                            # the strongest maps a picture of a stage shows
EYES = 120                           # the channel of layer3 that is strongest on the cat
DEAD_BELOW = 1e-5                    # a filter whose weights are all smaller than this does nothing

# The slider over the receptive field in *The idea*: what one number sees after conv1, the pooling and layer1 to layer4,
# for the number of each map nearest the middle of the cat's eye on the left. EYE is that eye in the photo of 224 by 224 that
# ResNet-18 sees (top, left, bottom, right, the last two included), read off the photo drawn with a grid of 10 pixels;
# filter 57, whose map is highest on the upper edge of that eye, is highest at pixel (56, 46).
FIELD_STAGES = ("conv1", "pooling", "layer1", "layer2", "layer3", "layer4")
FIELD_VIEWS = (7, POOLED, 43, 99, 211, 435)
FIELD_STRIDES = (2, 4, 4, 8, 16, 32)
FIELD_SIDES = (112, 56, 56, 28, 14, 7)
EYE = (55, 32, 112, 102)

# What the course measured with its code (course_feature_maps, everything_ai.models.resnet18 and its pinned file, the
# five steps of cv-prepare-photo, one thread). "zeros": the share of the numbers of each stage that are 0, on the cat.
# "strongest": the channels of a stage with the highest mean, strongest first. "eyes": channel EYES of layer3 on the
# cat: its largest number and where, and its rank by mean among the 256 on each photo. "pull": how much of the pull of
# the photo's pixels on its number at row 7 and column 7 of the cat (the sum over the colors of |gradient|, by
# autograd) comes from the middle square of each side around it; that number sees pixels 7 to 217, so its 211 by 211
# lie on the photo, and no edge cuts them off. "dead": the filters of conv1 whose weights are all below DEAD_BELOW;
# "largest": the largest |weight| of the other filters (smallest and largest), and of the filters the text names;
# "image": the darkest and the lightest number of filter 26 as an image. "highest": where a map of conv1 on the cat is
# highest; "quiet": the filters, not dead, whose map on a photo is 0 everywhere.
# tests/test_cv_inside_cnn.py measures it all again when EAI_RESNET18_FILE names the downloaded file.
REFERENCE = {
    "zeros": {"cat": [0.2935, 0.2342, 0.4558, 0.5284, 0.4663]},
    "strongest": {
        "cat": {"conv1": [28, 51, 60, 19, 52, 21, 61, 22], "layer1": [8, 50, 46, 61, 63, 11, 25, 47],
                "layer2": [58, 106, 82, 67, 41, 54, 36, 55], "layer3": [120, 159, 89, 116, 188, 53, 61, 206],
                "layer4": [461, 156, 491, 167, 225, 189, 493, 298]},
        "coffee": {"conv1": [28, 51, 19, 60, 22, 16, 6, 52], "layer3": [52, 89, 81, 3, 37, 15, 53, 192]},
        "fruit": {"conv1": [28, 6, 51, 19, 60, 22, 16, 61], "layer3": [33, 162, 52, 210, 193, 89, 202, 94]},
    },
    "eyes": {"max": 2.6034, "at": (6, 12), "rank": {"cat": 1, "coffee": 154, "fruit": 63}},
    "pull": {"at": (7, 7), 49: 0.3351, 97: 0.8164},
    "dead": [2, 4, 7, 9, 13, 36, 38, 48],
    "largest": {"alive": (0.0555, 1.0165), 9: 1.0849e-06, 17: 0.0757, 18: 0.2768, 26: 0.2763},
    "image": {26: (0.0094, 1.0)},
    "highest": {26: (62, 96), 28: (93, 73), 57: (28, 23)},
    "quiet": {"cat": [17, 18], "coffee": [], "fruit": [18]},
}


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def percent(share: float) -> int:
    return round(100 * share)


# --- the course's versions of the exercises ------------------------------------------------------------------------

def course_filter_image(w):
    """One filter of shape (3, k, k) as an image of shape (k, k, 3): gray, 0.5, for a weight of 0, and 0 or 1 for the
    largest weight of the filter, + or -."""
    return (0.5 + w / (2 * w.abs().max())).permute(1, 2, 0)


def course_feature_maps(model, x):
    """What each stage of a ResNet-18 gives for photos x of shape (n, 3, H, W): conv1, bn1 and a ReLU; then, after the
    max pooling, layer1 to layer4."""
    x = torch.relu(model.bn1(model.conv1(x)))
    maps = [x]
    x = nn.MaxPool2d(3, stride=2, padding=1)(x)
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        x = layer(x)
        maps.append(x)
    return maps


def crop(photo: str) -> np.ndarray:
    """The middle 224 by 224 of a photo after step 1 of cv-prepare-photo: what ResNet-18 sees, as colors of 0 to 255."""
    return _PREP.center(_PREP.resized(str(PHOTOS[photo])))


def prepared(photo: str):
    return _PREP.prepared(str(PHOTOS[photo]))


@functools.lru_cache(maxsize=4)
def course_maps(photo: str) -> tuple:
    """The course's maps of a photo, without the batch: one tensor per stage."""
    model = _TRY._resnet18()
    with torch.no_grad():
        return tuple(m[0] for m in course_feature_maps(model, prepared(photo)))


def strongest(maps, n: int = SHOWN) -> list[int]:
    """The n channels of one stage with the highest mean, strongest first."""
    return [int(k) for k in maps.mean(dim=(1, 2)).argsort(descending=True)[:n]]


def view(stage: int, row: int, col: int) -> tuple:
    """The square of the photo that the number at (row, col) of a map of a stage sees, clipped to the photo of 224 by
    224: (top, left, bottom, right), the last two included. Its middle is the pixel at stride · row, stride · col."""
    half = (VIEWS[stage] - 1) // 2
    r, c = STRIDES[stage] * row, STRIDES[stage] * col
    return max(r - half, 0), max(c - half, 0), min(r + half, 223), min(c + half, 223)


# --- the pictures ----------------------------------------------------------------------------------------------------

def _gray(m) -> np.ndarray:
    """A map as an image of 0 (black) to 1 (white, its largest number)."""
    m = np.asarray(m, dtype=float)
    top = m.max()
    return m / top if top > 0 else np.zeros_like(m)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _tiles(images: list, gap: int = 1) -> np.ndarray:
    """Images of k by k by 3 in a square grid, 8 to a row, with a transparent gap between them: one RGBA image."""
    k = images[0].shape[0]
    per_row = 8
    rows = math.ceil(len(images) / per_row)
    out = np.zeros((rows * (k + gap) - gap, per_row * (k + gap) - gap, 4))
    for i, img in enumerate(images):
        r, c = divmod(i, per_row)
        out[r * (k + gap):r * (k + gap) + k, c * (k + gap):c * (k + gap) + k, :3] = np.clip(img, 0, 1)
        out[r * (k + gap):r * (k + gap) + k, c * (k + gap):c * (k + gap) + k, 3] = 1
    return out


def filter_images(weight, filter_image=course_filter_image) -> list[np.ndarray]:
    """Every filter of a first layer as an image, with the given function."""
    return [np.asarray(torch.as_tensor(filter_image(weight[k].clone())), dtype=float) for k in range(weight.shape[0])]


def _slim(svg: str) -> str:
    """The images of a rendered figure, smaller, as the same pixels: a gray one as an 8-bit gray PNG, a large color one
    without transparency, such as a photo, as a JPEG; one with transparent parts stays as it is."""
    import base64
    import io
    from PIL import Image

    def one(found):
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(re.sub(r"\s", "", found.group(1))))).convert("RGBA"))
        if (img[..., 3] < 255).any():
            return found.group(0)
        rgb, buf = img[..., :3], io.BytesIO()
        if (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all():
            Image.fromarray(rgb[..., 0]).save(buf, "PNG", optimize=True)
            kind = "png"
        elif rgb.shape[0] * rgb.shape[1] >= 10_000:
            Image.fromarray(rgb).save(buf, "JPEG", quality=90)
            kind = "jpeg"
        else:
            Image.fromarray(rgb).save(buf, "PNG", optimize=True)
            kind = "png"
        return f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return re.sub(r"data:image/png;base64,([A-Za-z0-9+/=\s]+)", one, svg)


def grid_picture(images: list, labels: dict) -> str:
    """The filter grid: every filter of the first layer as a small color image, 8 to a row, with the number of the
    first filter of each row on its left and 0 to 7 along the top, so that filter k is in row k // 8, column k % 8.
    `labels` gives "title" and "alt"."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    k = images[0].shape[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    ax.imshow(_tiles(images), interpolation="none")
    ax.set_xticks([c * (k + 1) + (k - 1) / 2 for c in range(8)], [str(c) for c in range(8)])
    ax.set_yticks([r * (k + 1) + (k - 1) / 2 for r in range(math.ceil(len(images) / 8))],
                  [str(8 * r) for r in range(math.ceil(len(images) / 8))])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=9, pad=2)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(labels["title"], fontsize=10)
    out = _slim(plots.render(fig, alt=labels["alt"]))
    plt.close(fig)
    return out


def filter_frames(weight, maps, labels: dict) -> dict:
    """For the slider over the filters: for every filter k of conv1, filter k large, and its feature map on the cat
    after conv1, bn1 and the ReLU (black 0, white its largest number), as HTML. `labels` gives "filter" and "map"
    ({k}), and "alt" ({k}). One figure, drawn again for every filter."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    images = filter_images(weight)
    fig, (one, fmap) = plt.subplots(1, 2, figsize=(4.6, 2.5), gridspec_kw={"wspace": 0.12})
    shown = one.imshow(images[0], interpolation="none")
    mapped = fmap.imshow(_gray(maps[0]), cmap="gray", vmin=0, vmax=1, interpolation="none")
    for ax in (one, fmap):
        _bare(ax)
    t_one, t_map = one.set_title("", fontsize=9), fmap.set_title("", fontsize=9)
    frames = {}
    for k in range(weight.shape[0]):
        shown.set_data(images[k])
        mapped.set_data(_gray(maps[k]))
        t_one.set_text(labels["filter"].format(k=k))
        t_map.set_text(labels["map"].format(k=k))
        frames[k] = _slim(plots.render(fig, alt=labels["alt"].format(k=k)))
    plt.close(fig)
    return frames


def stage_frames(maps: list, photo: np.ndarray, labels: dict) -> dict:
    """For the slider over the stages: for every stage, the photo with the square its strongest number sees, and the
    SHOWN strongest maps of that stage (black 0, white the largest number of each), the strongest first, with a frame
    around that number. `maps` are the five stages without the batch. `labels` gives "photo", "map" ({k}) and "alt"
    ({stage})."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from everything_ai import plots
    frames = {}
    for i, m in enumerate(maps):
        order = strongest(m)
        fig = plt.figure(figsize=(6.6, 2.75))
        outer = fig.add_gridspec(1, 2, width_ratios=[1, 2.15], wspace=0.06)
        ax = fig.add_subplot(outer[0])
        ax.imshow(photo, interpolation="none")
        _bare(ax)
        ax.set_title(labels["photo"], fontsize=9)
        r, c = divmod(int(m[order[0]].argmax()), m.shape[2])
        top, left, bottom, right = view(i, r, c)
        for color, width in ((plots.PAGE, 3.4), (plots.INK, 1.6)):
            ax.add_patch(Rectangle((left - 0.5, top - 0.5), right - left + 1, bottom - top + 1, fill=False, edgecolor=color, linewidth=width))
        inner = outer[1].subgridspec(2, 4, wspace=0.08, hspace=0.32)
        for j, k in enumerate(order):
            a = fig.add_subplot(inner[divmod(j, 4)])
            a.imshow(_gray(m[k]), cmap="gray", vmin=0, vmax=1, interpolation="none")
            _bare(a)
            a.set_title(labels["map"].format(k=k), fontsize=8.5)
            if j == 0:
                for color, width in ((plots.PAGE, 2.6), (plots.INK, 1.2)):
                    a.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=width))
        frames[i] = _slim(plots.render(fig, alt=labels["alt"].format(stage=STAGES[i])))
        plt.close(fig)
    return frames


def field_square(i: int) -> tuple:
    """For stop i of the slider over the receptive field: the row and column of the number of the map nearest the
    middle of EYE, and the square of the photo it sees, (top, left, bottom, right), not clipped to the photo."""
    stride, half = FIELD_STRIDES[i], (FIELD_VIEWS[i] - 1) // 2
    r = min(int((EYE[0] + EYE[2]) / 2 / stride + 0.5), FIELD_SIDES[i] - 1)
    c = min(int((EYE[1] + EYE[3]) / 2 / stride + 0.5), FIELD_SIDES[i] - 1)
    return r, c, (stride * r - half, stride * c - half, stride * r + half, stride * c + half)


def sees_eye(i: int) -> bool:
    top, left, bottom, right = field_square(i)[2]
    return top <= EYE[0] and left <= EYE[1] and bottom >= EYE[2] and right >= EYE[3]


def field_frames(labels: dict) -> dict:
    """For the slider over the receptive field: for every stop, the cat as ResNet-18 sees it, the squares of every
    stage thin, the square of this stage bold, and the eye on the left dashed; the squares that pass the edge of the photo go
    on over the page around it. The value of a stop is the side of its square, as FIELD_VIEWS. `labels` gives "alt"
    ({stage})."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from everything_ai import plots
    photo = crop("cat")
    squares = [field_square(i)[2] for i in range(len(FIELD_VIEWS))]
    frames = {}
    for i, view_ in enumerate(FIELD_VIEWS):
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.imshow(photo, extent=(-0.5, 223.5, 223.5, -0.5), interpolation="none")
        ax.add_patch(Rectangle((-0.5, -0.5), 224, 224, fill=False, edgecolor=plots.INK, linewidth=0.6))
        ax.add_patch(Rectangle((EYE[1] - 0.5, EYE[0] - 0.5), EYE[3] - EYE[1] + 1, EYE[2] - EYE[0] + 1, fill=False,
                               edgecolor=plots.PAGE, linewidth=1.3, linestyle=(0, (3, 2))))
        for j, (top, left, bottom, right) in enumerate(squares):
            ax.add_patch(Rectangle((left - 0.5, top - 0.5), right - left + 1, bottom - top + 1, fill=False,
                                   edgecolor=plots.INK, linewidth=0.7, alpha=0.35))
        top, left, bottom, right = squares[i]
        for color, width in ((plots.PAGE, 3.4), (plots.INK, 1.7)):
            ax.add_patch(Rectangle((left - 0.5, top - 0.5), right - left + 1, bottom - top + 1, fill=False, edgecolor=color, linewidth=width))
        ax.set_xlim(-160, 290)
        ax.set_ylim(320, -130)
        _bare(ax)
        for side in ax.spines.values():   # no frame: the thin line around the photo shows where it ends
            side.set_color("none")
        frames[view_] = _slim(plots.render(fig, alt=labels["alt"].format(stage=FIELD_STAGES[i])))
        plt.close(fig)
    return frames


def field_sentence(view_: int, lang: str) -> str:
    """What the slider over the receptive field says under its picture."""
    i = FIELD_VIEWS.index(view_)
    r, c, (top, left, bottom, right) = field_square(i)
    side, stride = FIELD_SIDES[i], FIELD_STRIDES[i]
    middle = (stride * r, stride * c)
    whole = top <= 0 and left <= 0 and bottom >= 223 and right >= 223
    outside = top < 0 or left < 0 or bottom > 223 or right > 223
    if lang == "nl":
        stage = "de pooling" if FIELD_STAGES[i] == "pooling" else f"`{FIELD_STAGES[i]}`"
        said = (f"Na {stage}: maps van {side} bij {side}, met {stride} pixels van de foto tussen buurgetallen. Het getal op rij {r} "
                f"en kolom {c} ziet {view_} bij {view_} pixels rond rij {middle[0]} en kolom {middle[1]} van de foto: het dikke vierkant. ")
        said += "Het hele oog links past erin." if sees_eye(i) else "Het oog links, het gestippelde vak, past er nog niet in."
        if whole:
            said += " Dat is meer dan de hele foto, die er 224 bij 224 van vult; de rest is padding."
        elif outside:
            said += " Een deel ligt buiten de foto: daar ziet het padding."
        return said
    stage = "the pooling" if FIELD_STAGES[i] == "pooling" else f"`{FIELD_STAGES[i]}`"
    said = (f"After {stage}: maps of {side} by {side}, with {stride} pixels of the photo between neighboring numbers. The number at "
            f"row {r} and column {c} sees {view_} by {view_} pixels around row {middle[0]} and column {middle[1]} of the photo: the bold square. ")
    said += "The whole eye on the left fits in it." if sees_eye(i) else "The eye on the left, the dashed box, does not fit in it yet."
    if whole:
        said += " That is more than the whole photo, which fills 224 by 224 of it; the rest is padding."
    elif outside:
        said += " Part of it lies outside the photo: there it sees padding."
    return said


def stage_sentence(maps: list, i: int, lang: str) -> str:
    """What the slider over the stages says under its picture."""
    m = maps[i]
    order = strongest(m)
    r, c = divmod(int(m[order[0]].argmax()), m.shape[2])
    zeros = percent(float((m == 0).float().mean()))
    channels, side = m.shape[0], m.shape[1]
    top, left, bottom, right = view(i, r, c)
    whole = (top, left, bottom, right) == (0, 0, 223, 223)
    cut = bottom - top + 1 < VIEWS[i] or right - left + 1 < VIEWS[i]   # part of the piece lies outside the photo
    if lang == "nl":
        wide = (f"ziet de hele foto, en meer: een stuk van {VIEWS[i]} bij {VIEWS[i]} pixels, waarvan de foto er 224 bij 224 vult"
                if whole else f"ziet een stuk van {VIEWS[i]} bij {VIEWS[i]} pixels: het vierkant{', voor zover het op de foto ligt' if cut else ''}")
        return (f"Na **{STAGES[i]}**: {channels} maps van {side} bij {side}, en {zeros}% van hun getallen is 0. "
                f"De sterkste, channel {order[0]}, is het hoogst op rij {r} en kolom {c}; dat getal {wide}.")
    wide = (f"sees the whole photo, and more: a piece of {VIEWS[i]} by {VIEWS[i]} pixels, of which the photo fills 224 by 224"
            if whole else f"sees a piece of {VIEWS[i]} by {VIEWS[i]} pixels: the square{', as far as it lies on the photo' if cut else ''}")
    return (f"After **{STAGES[i]}**: {channels} maps of {side} by {side}, and {zeros}% of their numbers are 0. "
            f"The strongest, channel {order[0]}, is highest at row {r} and column {c}; that number {wide}.")


def filter_sentence(maps, weight, k: int, lang: str) -> str:
    """What the slider over the filters says under its picture: the largest weight of filter k, + or -, how much of its
    map on the cat is 0, and where that map is highest."""
    m, size = maps[k], float(weight[k].abs().max())
    nl = lang == "nl"
    if size < DEAD_BELOW:
        return (f"Filter {k}: zijn grootste weight is kleiner dan 0,00001, en zijn map op de kat is overal 0: dit filter doet niets." if nl else
                f"Filter {k}: its largest weight is below 0.00001, and its map on the cat is 0 everywhere: this filter does nothing.")
    if float(m.max()) <= 0:
        return (f"Filter {k}: zijn grootste weight is {number(size, 2, lang)}, en zijn map op de kat is overal 0." if nl else
                f"Filter {k}: its largest weight is {number(size, 2, lang)}, and its map on the cat is 0 everywhere.")
    share = float((m == 0).float().mean())
    r, c = divmod(int(m.argmax()), m.shape[1])
    if nl:
        zeros = ("geen getal van zijn map op de kat is 0" if share == 0 else
                 "minder dan 1% van zijn map op de kat is 0" if share < 0.005 else f"{percent(share)}% van zijn map op de kat is 0")
        return (f"Filter {k}: zijn grootste weight is {number(size, 2, lang)}; {zeros}, en die map is het hoogst "
                f"op rij {r} en kolom {c}, met {number(float(m.max()), 2, lang)}.")
    zeros = ("no number of its map on the cat is 0" if share == 0 else
             "less than 1% of its map on the cat is 0" if share < 0.005 else f"{percent(share)}% of its map on the cat is 0")
    return (f"Filter {k}: its largest weight is {number(size, 2, lang)}; {zeros}, and that map is highest "
            f"at row {r} and column {c}, at {number(float(m.max()), 2, lang)}.")


# --- the exercises ---------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "inside-shape" and "eyes" are the choice questions in the lesson.
REQUIRED = ["inside-shape", "filter-image", "feature-maps", "eyes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _raised_in(error: BaseException, name: str) -> bool:
    """The error happened inside the function of that name: in the learner's own code, not in the check's lines."""
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == name:
            return True
        tb = tb.tb_next
    return False


def _close(got, want) -> bool:
    """The same value, with floats close; nan and inf count as themselves, so that two wrong versions can match."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return False
        if math.isnan(want) or math.isinf(want):
            return (math.isnan(got) and math.isnan(want)) or got == want
        return abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

# --- exercise 2: a filter as an image ------------------------------------------------------------------------------

_F_HEAD = "import torch\n\n\ndef filter_image(w):\n"
_F_START = (_F_HEAD + "    # w: the weights of one filter, of shape (3, 7, 7): a 7 by 7 for each color\n"
            "    # give back an image of shape (7, 7, 3) with numbers from 0 to 1: gray, 0.5,\n"
            "    # for a weight of 0, and 0 or 1 for the largest weight of the filter, + or -\n    ...\n")
_F_BODY = "    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 2, 0)\n"


def _f(body: str = _F_BODY) -> str:
    return _F_HEAD + body


_F = _f()
_F_SCENARIOS = [
    # a made-up filter of 3 by 7 by 7: the shape, the darkest and the lightest number, and the red of row 0, column 6
    "torch.manual_seed(0)\nw = torch.randn(3, 7, 7)\nimg = filter_image(w)\n"
    "[list(img.shape), img.min().item(), img.max().item(), img[0, 6, 0].item()]",
    # a filter of zeros with one weight of -2, green in the middle: a pixel at the edge and the one in the middle
    "w = torch.zeros(3, 7, 7)\nw[1, 3, 3] = -2.0\nimg = filter_image(w)\n[img[0, 0].tolist(), img[3, 3].tolist()]",
    # a filter of another size, 3 by 3, with the weights -13 to 13: its top row
    "w = torch.arange(-13, 14).float().reshape(3, 3, 3)\nimg = filter_image(w)\n[list(img.shape), img[0].tolist()]",
]
_F_WRONG = {
    "no-permute": _f("    m = w.abs().max()\n    return 0.5 + w / (2 * m)\n"),
    "transposed": _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(2, 1, 0)\n"),
    "max-only": _f("    m = w.max()\n    return (0.5 + w / (2 * m)).permute(1, 2, 0)\n"),
    "min-max": _f("    w = (w - w.min()) / (w.max() - w.min())\n    return w.permute(1, 2, 0)\n"),
    "reshape": _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).reshape(7, 7, 3)\n"),
    "no-half": _f("    m = w.abs().max()\n    return (w / (2 * m)).permute(1, 2, 0)\n"),
    "no-two": _f("    m = w.abs().max()\n    return (0.5 + w / m).permute(1, 2, 0)\n"),
}

# --- exercise 3: the feature maps of every stage -------------------------------------------------------------------

_M_HEAD = "import torch\nfrom torch import nn\n\n\ndef feature_maps(model, x):\n"
_M_START = (_M_HEAD + "    # model: a ResNet-18; x: photos of shape (n, 3, H, W), prepared as for ImageNet\n"
            "    # give back a list of 5 tensors, what each stage gives:\n"
            "    # 1. conv1, bn1 and a ReLU (then comes max pooling: 3 by 3, stride 2, padding 1)\n"
            "    # 2. to 5. layer1, layer2, layer3 and layer4\n    ...\n")
_M_BODY = ("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2, padding=1)(x)\n"
           "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n"
           "    return maps\n")


def _mm(body: str = _M_BODY) -> str:
    return _M_HEAD + body


_M = _mm()
_M_MODEL = ("from everything_ai.models import ResNet18\n\ntorch.manual_seed(0)\n"
            "model = ResNet18().eval()       # random weights, not trained: the check needs no download\n"
            "model.bn1.running_var.fill_(4)  # bn1 now divides by 2, so that it shows\n"
            "x = torch.randn(1, 3, 64, 64)\nmaps = feature_maps(model, x)\n")
_M_SCENARIOS = [
    # the shape of each map: a photo of 64 by 64 gives maps of 32, 16, 8, 4 and 2
    _M_MODEL + "[list(m.shape) for m in maps]",
    # the first stage: its smallest number (0 after the ReLU) and its mean, and the mean of the second
    _M_MODEL + "[maps[0].min().item(), maps[0].mean().item(), maps[1].mean().item()]",
    # the last maps, averaged over their rows and columns, are what ResNet-18 gives its last layer
    _M_MODEL + "(maps[4].mean(dim=(2, 3)) - model.features(x)).abs().max().item()",
]
_M_WRONG = {
    "no-relu": _mm("    x = model.bn1(model.conv1(x))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2, padding=1)(torch.relu(x))\n"
                   "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"),
    "no-bn": _mm("    x = torch.relu(model.conv1(x))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2, padding=1)(x)\n"
                 "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"),
    "no-pool": _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n"
                   "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"),
    "pool-first": _mm("    x = nn.MaxPool2d(3, stride=2, padding=1)(torch.relu(model.bn1(model.conv1(x))))\n    maps = [x]\n"
                      "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"),
    "no-padding": _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2)(x)\n"
                      "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"),
    "four": _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    x = nn.MaxPool2d(3, stride=2, padding=1)(x)\n    maps = []\n"
                "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"),
}


EXERCISES = {
    "filter-image": ScenarioBox(
        name="filter-image",
        start=_F_START,
        solution=_F,
        scenarios=_F_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "filters"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the `.shape` of what your function gives back, and it gives back nothing, `None`: end with `return` and the image.",
                    nl="De check vraagt de `.shape` van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en het plaatje."),
            Mistake("permute-two", error=lambda e: isinstance(e, RuntimeError) and "permute" in str(e) and "number of dimensions" in str(e),
                    en="`.permute` needs the new place of every axis, and a filter has 3: the colors, the rows and the columns. `.permute(1, 2, 0)` puts axis 1 first, then axis 2, then axis 0, the colors.",
                    nl="`.permute` wil de nieuwe plek van elke axis, en een filter heeft er 3: de kleuren, de rijen en de kolommen. `.permute(1, 2, 0)` zet axis 1 vooraan, dan axis 2, dan axis 0, de kleuren."),
            Mistake("shape", error=lambda e: isinstance(e, IndexError) and not _raised_in(e, "filter_image"),
                    en="The check asks for the pixel at row 0 and column 6 of your image, and it is not there: your image has another shape than (7, 7, 3). Put the rows first, then the columns, then the colors: `.permute(1, 2, 0)`.",
                    nl="De check vraagt de pixel op rij 0 en kolom 6 van je plaatje, en die is er niet: je plaatje heeft een andere shape dan (7, 7, 3). Zet de rijen vooraan, dan de kolommen, dan de kleuren: `.permute(1, 2, 0)`."),
            Mistake("no-permute", wrong=_F_WRONG["no-permute"],
                    en=_S_EN + ": your image still has the colors first, shape (3, 7, 7). `plt.imshow` wants an image as rows, columns and colors: `.permute(1, 2, 0)` puts axis 1 first, then axis 2, then the colors.",
                    nl=_S_NL + ": je plaatje heeft de kleuren nog vooraan, shape (3, 7, 7). `plt.imshow` wil een plaatje als rijen, kolommen en kleuren: `.permute(1, 2, 0)` zet axis 1 vooraan, dan axis 2, dan de kleuren."),
            Mistake("transposed", wrong=_F_WRONG["transposed"],
                    en=_S_EN + ": the shape is right, but rows and columns changed places, so the filter is mirrored along its diagonal: a horizontal edge would show as a vertical one. `.permute(2, 1, 0)` puts the columns first; `.permute(1, 2, 0)` keeps the rows first.",
                    nl=_S_NL + ": de shape klopt, maar rijen en kolommen zijn van plaats gewisseld, dus is het filter gespiegeld langs zijn diagonaal: een horizontale rand zou er als een verticale uitzien. `.permute(2, 1, 0)` zet de kolommen vooraan; `.permute(1, 2, 0)` houdt de rijen vooraan."),
            Mistake("max-only", wrong=_F_WRONG["max-only"],
                    en=_S_EN + ": you divide by `w.max()`, the largest weight with its sign, and that can be negative, or 0, as in the second filter. Divide by the largest weight, + or -: `w.abs().max()`.",
                    nl=_S_NL + ": je deelt door `w.max()`, de grootste weight met zijn teken, en die kan negatief zijn, of 0, zoals in het tweede filter. Deel door de grootste weight, + of -: `w.abs().max()`."),
            Mistake("reshape", matches=lambda got, text: re.search(r"\.(reshape|view)\(", text) is not None and re.search(r"\.(permute|movedim|transpose)\(|\.T\b", text) is None,
                    en=_S_EN + ": the shape is right, but the numbers are in the wrong places. `reshape` keeps the numbers in their order and only cuts them anew: the first pixel gets the first three weights of red. `.permute(1, 2, 0)` moves the axes themselves.",
                    nl=_S_NL + ": de shape klopt, maar de getallen staan op de verkeerde plekken. `reshape` houdt de getallen in hun volgorde en knipt ze alleen opnieuw: de eerste pixel krijgt de eerste drie weights van rood. `.permute(1, 2, 0)` verplaatst de assen zelf."),
            Mistake("min-max", wrong=_F_WRONG["min-max"],
                    en=_S_EN + ": from the smallest to the largest weight, the image does go from 0 to 1, but a weight of 0 is no longer gray, 0.5: in the second filter, the zeros are white. Keep 0 at 0.5, with `0.5 + w / (2 * m)` and `m` the largest weight, + or -.",
                    nl=_S_NL + ": van de kleinste naar de grootste weight gaat het plaatje wel van 0 tot 1, maar een weight van 0 is niet meer grijs, 0,5: in het tweede filter zijn de nullen wit. Houd 0 op 0,5, met `0.5 + w / (2 * m)` en `m` de grootste weight, + of -."),
            Mistake("no-half", wrong=_F_WRONG["no-half"],
                    en=_S_EN + ": your numbers go from -0.5 to 0.5, so a weight of 0 is black and every negative weight is below black. Add 0.5: then 0 is gray, in the middle of 0 and 1.",
                    nl=_S_NL + ": je getallen gaan van -0,5 tot 0,5, dus is een weight van 0 zwart en elke negatieve weight donkerder dan zwart. Tel er 0,5 bij op: dan is 0 grijs, midden tussen 0 en 1."),
            Mistake("no-two", wrong=_F_WRONG["no-two"],
                    en=_S_EN + ": `w / m` goes from -1 to 1, so with 0.5 added, your numbers go from -0.5 to 1.5. Divide by `2 * m`: then they go from 0 to 1.",
                    nl=_S_NL + ": `w / m` gaat van -1 tot 1, dus met 0,5 erbij gaan je getallen van -0,5 tot 1,5. Deel door `2 * m`: dan gaan ze van 0 tot 1."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: scale the weights so that 0 is gray and the largest is 0 or 1, and put the colors last.",
                      "nl": "Vervang de `...` door je code: schaal de weights zo dat 0 grijs is en de grootste 0 of 1, en zet de kleuren achteraan."},
            "wrong": {"en": _S_EN + ": the shape, the darkest and the lightest number, and the red of the pixel at row 0, column 6. Is a weight of 0 at 0.5, does the largest weight, + or -, end at 0 or 1, and are the colors last?",
                      "nl": _S_NL + ": de shape, het donkerste en het lichtste getal, en het rood van de pixel op rij 0, kolom 6. Staat een weight van 0 op 0,5, komt de grootste weight, + of -, op 0 of 1 uit, en staan de kleuren achteraan?"},
            "almost": {"en": "Right for the first filter. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any filter, also one whose largest weight is negative, and one of another size?",
                       "nl": "Goed voor het eerste filter. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk filter, ook een waarvan de grootste weight negatief is, en een van een andere maat?"},
            "correct": {"en": "Your `filter_image` is right: a weight of 0 is gray, positive weights are lighter in their color, negative ones darker. It is now in your project, as `{saved}`.",
                        "nl": "Je `filter_image` klopt: een weight van 0 is grijs, positieve weights zijn lichter in hun kleur, negatieve donkerder. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `filter_image` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `filter_image` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "feature-maps": ScenarioBox(
        name="feature-maps",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "maps"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, TypeError) and "'NoneType' object is not iterable" in str(e),
                    en="The check goes through what your function gives back, and it gives back nothing, `None`: end with `return` and the list of maps.",
                    nl="De check loopt langs wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de lijst met maps."),
            Mistake("torchvision", error=lambda e: isinstance(e, AttributeError) and re.search(r"has no attribute '(maxpool|relu|avgpool)'", str(e)) is not None,
                    en="Unlike torchvision's, the course's ResNet-18 has no module for its pooling or its ReLU: it calls them as functions. Make the pooling yourself, `nn.MaxPool2d(3, stride=2, padding=1)`, and the ReLU with `torch.relu`.",
                    nl="Anders dan die van torchvision heeft de ResNet-18 van de course geen module voor zijn pooling of zijn ReLU: hij roept ze aan als functies. Maak de pooling zelf, `nn.MaxPool2d(3, stride=2, padding=1)`, en de ReLU met `torch.relu`."),
            Mistake("no-relu", wrong=_M_WRONG["no-relu"],
                    en=_S_EN + ": the first maps have negative numbers, so they come before the ReLU. Keep what `torch.relu` gives after `bn1`: a map of a stage is what comes out of it, ReLU included.",
                    nl=_S_NL + ": de eerste maps hebben negatieve getallen, dus komen ze van voor de ReLU. Bewaar wat `torch.relu` na `bn1` geeft: een map van een fase is wat eruit komt, de ReLU erbij."),
            Mistake("no-bn", wrong=_M_WRONG["no-bn"],
                    en=_S_EN + ": the first maps have twice the mean they should have, because `bn1` is missing; in this ResNet-18, it divides by 2. The first stage is `conv1`, then `bn1`, then the ReLU.",
                    nl=_S_NL + ": de eerste maps hebben twee keer het gemiddelde dat ze moeten hebben, want `bn1` ontbreekt; in deze ResNet-18 deelt hij door 2. De eerste fase is `conv1`, dan `bn1`, dan de ReLU."),
            Mistake("no-pool", wrong=_M_WRONG["no-pool"],
                    en=_S_EN + ": from `layer1` on, the maps have twice as many rows and columns as they should, because the max pooling between the first stage and `layer1` is missing: `nn.MaxPool2d(3, stride=2, padding=1)`.",
                    nl=_S_NL + ": vanaf `layer1` hebben de maps twee keer zoveel rijen en kolommen als ze moeten hebben, want de max pooling tussen de eerste fase en `layer1` ontbreekt: `nn.MaxPool2d(3, stride=2, padding=1)`."),
            Mistake("no-padding", wrong=_M_WRONG["no-padding"],
                    en=_S_EN + ": `layer1` gives maps of 15 by 15 instead of 16 by 16, because your max pooling has no padding. With `padding=1`, a pooling of 3 by 3 with stride 2 halves the rows and the columns exactly: `nn.MaxPool2d(3, stride=2, padding=1)`.",
                    nl=_S_NL + ": `layer1` geeft maps van 15 bij 15 in plaats van 16 bij 16, want je max pooling heeft geen padding. Met `padding=1` halveert een pooling van 3 bij 3 met stride 2 de rijen en de kolommen precies: `nn.MaxPool2d(3, stride=2, padding=1)`."),
            Mistake("pool-first", wrong=_M_WRONG["pool-first"],
                    en=_S_EN + ": your first maps come after the max pooling, so they have half the rows and columns. Keep the first stage's maps before the pooling, and pool what goes on to `layer1`.",
                    nl=_S_NL + ": je eerste maps komen na de max pooling, dus hebben ze half zoveel rijen en kolommen. Bewaar de maps van de eerste fase vóór de pooling, en pool wat doorgaat naar `layer1`."),
            Mistake("four", wrong=_M_WRONG["four"],
                    en=_S_EN + ": your list has 4 tensors, from `layer1` to `layer4`. The first stage counts too: keep what conv1, bn1 and the ReLU give, before the pooling.",
                    nl=_S_NL + ": je lijst heeft 4 tensors, van `layer1` tot en met `layer4`. De eerste fase telt ook mee: bewaar wat conv1, bn1 en de ReLU geven, vóór de pooling."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and re.search(r"expected input\[.*\] to have \d+ channels", str(e)) is not None,
                    en="A layer gets another number of channels than it was made for: each stage gets what the one before it gives, in the order conv1, bn1, the ReLU, the pooling, then `layer1`, `layer2`, `layer3` and `layer4`.",
                    nl="Een layer krijgt een ander aantal channels dan waarvoor hij gemaakt is: elke fase krijgt wat de vorige geeft, in de volgorde conv1, bn1, de ReLU, de pooling, dan `layer1`, `layer2`, `layer3` en `layer4`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the first stage, the pooling, then layer1 to layer4, keeping what each stage gives in a list.",
                      "nl": "Vervang de `...` door je code: de eerste fase, de pooling, dan layer1 tot en met layer4, en bewaar wat elke fase geeft in een lijst."},
            "wrong": {"en": _S_EN + ": the shape of each of the 5 maps. Does your list hold what the first stage gives, before the pooling, and then what each of `layer1` to `layer4` gives?",
                      "nl": _S_NL + ": de shape van elk van de 5 maps. Bevat je lijst wat de eerste fase geeft, vóór de pooling, en dan wat elk van `layer1` tot en met `layer4` geeft?"},
            "almost": {"en": "The shapes are right. But " + _S_EN[0].lower() + _S_EN[1:] + ". Is every stage complete: conv1, bn1 and the ReLU, and each layer on what the one before it gives?",
                       "nl": "De shapes kloppen. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Is elke fase compleet: conv1, bn1 en de ReLU, en elke layer op wat de vorige geeft?"},
            "correct": {"en": "Your `feature_maps` is right: the averages of the last maps over their rows and columns are exactly what ResNet-18 gives its last layer. It is now in your project, as `{saved}`.",
                        "nl": "Je `feature_maps` klopt: de gemiddelden van de laatste maps over hun rijen en kolommen zijn precies wat ResNet-18 aan zijn laatste layer geeft. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `feature_maps` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `feature_maps` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("filter-image", _F_START, "empty", None),
    ("filter-image", _F, "correct", None),
    ("filter-image", _f("    return (w / w.abs().max() + 1).permute(1, 2, 0) / 2\n"), "correct", None),
    ("filter-image", _f("    img = w.permute(1, 2, 0)\n    return 0.5 + 0.5 * img / img.abs().max()\n"), "correct", None),
    ("filter-image", _f("    m = max(w.max(), -w.min())\n    return (0.5 + w / (2 * m)).permute(1, 2, 0)\n"), "correct", None),
    # a NumPy array works as well as a tensor
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 2, 0).numpy()\n"), "correct", None),
    ("filter-image", _f("    m = w.abs().max()\n    (0.5 + w / (2 * m)).permute(1, 2, 0)\n"), "error", "no-return"),
    ("filter-image", _f("    m = w.abs().max()\n    return ...\n"), "wrong", "dots-left"),
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 0, 2)\n"), "error", "shape"),
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 2)\n"), "error", "permute-two"),
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).view(7, 7, 3)\n"), "wrong", "reshape"),
    # the same numbers moved with reshape and .T: right
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).reshape(3, -1).T.reshape(w.shape[1], w.shape[2], 3)\n"), "correct", None),
    # an IndexError in the learner's own lines is not about the shape of the image
    ("filter-image", _f("    m = w[3].abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 2, 0)\n"), "error", None),
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 2, 0).clamp(0.1, 1)\n"), "almost", None),
    ("filter-image", _f("    m = w.abs().max()\n    return (0.5 + w / (2 * m)).permute(1, 2, 0) * 0.9\n"), "wrong", None),
    ("filter-image", _f("    m = w.abs().max()\n    if m > 2:\n        return (0.5 + w / (2 * m)).permute(1, 2, 0)\n    return w.permute(1, 2, 0)\n"), "almost", None),
] + [("filter-image", code, "wrong", key) for key, code in _F_WRONG.items()] + [
    ("feature-maps", _M_START, "empty", None),
    ("feature-maps", _M, "correct", None),
    ("feature-maps", _mm("    pool = nn.MaxPool2d(3, stride=2, padding=1)\n    a = torch.relu(model.bn1(model.conv1(x)))\n    b = model.layer1(pool(a))\n"
                         "    c = model.layer2(b)\n    d = model.layer3(c)\n    return [a, b, c, d, model.layer4(d)]\n"), "correct", None),
    ("feature-maps", _mm("    x = nn.ReLU()(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)\n"
                         "    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):\n        x = stage(x)\n        maps = maps + [x]\n    return maps\n"), "correct", None),
    # without gradients, with eval() called once more: the same maps
    ("feature-maps", _mm("    model.eval()\n    with torch.no_grad():\n" + "".join("    " + line + "\n" for line in _M_BODY.splitlines())), "correct", None),
    ("feature-maps", _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2, padding=1)(x)\n"
                         "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n"), "error", "no-return"),
    ("feature-maps", _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = model.maxpool(x)\n"
                         "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"), "error", "torchvision"),
    ("feature-maps", _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2, padding=1)(x)\n"
                         "    for layer in [model.layer2, model.layer1, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"), "error", "sizes"),
    # every layer on the pooled first maps: layer3 gets 64 channels instead of 128
    ("feature-maps", _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(3, stride=2, padding=1)(x)\n"
                         "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        maps.append(layer(x))\n    return maps\n"), "error", "sizes"),
    ("feature-maps", _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    ...\n    return maps\n"), "wrong", "dots-left"),
    ("feature-maps", _mm("    x = torch.relu(model.bn1(model.conv1(x)))\n    maps = [x]\n    x = nn.MaxPool2d(2)(x)\n"
                         "    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:\n        x = layer(x)\n        maps.append(x)\n    return maps\n"), "almost", None),
] + [("feature-maps", code, "wrong", key) for key, code in _M_WRONG.items()]


# --- the demo: the learner's functions on ResNet-18 ------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def demo(photo: str):
    """The demo's result for a photo, worked out anew each time, in about a second: None when filters.py or maps.py is not in the project yet; "filters" or "maps" when the learner's
    filter_image or feature_maps stops with an error or gives something of another shape; else a dict with the
    filter images ("images"), the five maps of the photo without the batch ("maps") and the photo as ResNet-18 sees
    it ("photo")."""
    if not (_saved("filters") and _saved("maps")):
        return None
    run = run_code("from cnn.filters import filter_image")
    if run.error is not None:
        return "filters"
    filter_image = run.names["filter_image"]
    run = run_code("from cnn.maps import feature_maps")
    if run.error is not None:
        return "maps"
    feature_maps = run.names["feature_maps"]
    model = _TRY._resnet18()
    weight = model.conv1.weight.detach()
    try:
        images = with_time_limit(lambda: filter_images(weight, filter_image), 30)
        if any(img.shape != (7, 7, 3) or not np.isfinite(img).all() for img in images):
            return "filters"
    except Exception:
        return "filters"
    try:
        with torch.no_grad():
            maps = with_time_limit(lambda: list(feature_maps(model, prepared(photo))), 60)
        maps = [torch.as_tensor(m).detach()[0] for m in maps]
        if [tuple(m.shape) for m in maps] != [(c, s, s) for c, s in zip(CHANNELS, SIDES)] or not all(torch.isfinite(m).all() for m in maps):
            return "maps"
    except Exception:
        return "maps"
    return {"images": images, "maps": maps, "photo": crop(photo)}
