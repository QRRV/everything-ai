# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-saliency: saliency, the saliency map of a photo for a class (per pixel the largest |slope|
of the class's logit over the three colors, by one backward pass), and most_salient, the share of the pixels with the
highest saliency; each saved in the learner's project (work/projects/cnn/saliency.py and salient.py) when right. Both
are checked on a ResNet-18 with random weights and on made-up maps, so they work without the download. ResNet-18
itself, with torchvision's weights trained on ImageNet, 47 MB, the lesson downloads once when the learner asks for it,
as cv-try-classifier does: the worked example, the maps of the three photos and the demo need it. The slider in *The
idea* needs no download: it shows probabilities the course measured, in REFERENCE. tests/test_cv_saliency.py measures
them all again when EAI_RESNET18_FILE names the downloaded file."""
import html
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import models, paths, projects
from everything_ai.check import Mistake, ScenarioBox, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_saliency",
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
FRUIT_LABELS = paths.ROOT / "data" / "photos" / "fruit-labels.png"   # per pixel: 0 tablecloth, 1 apple, 2 orange, 3 lemon, 4 lime

# The classes the lesson names, by their place among ImageNet's 1,000 logits (data/imagenet-classes.txt, from 0).
TIGER, EGYPTIAN, ESPRESSO, BALLOON, PING_PONG, ORANGE = 282, 285, 967, 417, 722, 950
TOP = {"cat": EGYPTIAN, "coffee": ESPRESSO, "fruit": BALLOON}   # ResNet-18's best guess for each photo
SIDE = 224            # the photo as ResNet-18 sees it: 224 by 224 pixels
SHARE = 0.05          # the share of the pixels that the slider and the demo change: 2,509 of the 50,176
WHITE = 0.99          # a map is drawn white from this quantile on: the 1% largest
STEPS = list(range(13))   # the stops of the slider: how much each color of a pixel changes, on its scale of 0 to 255
SEED = 0              # the random pixels: np.random.default_rng(SEED).permutation of the 50,176

# What the course measured with its code (course_saliency and course_gradient, everything_ai.models.resnet18 and its
# pinned file, the five steps of cv-prepare-photo, one thread). Probabilities are softmax over the 1,000 logits.
# "logit": the logit of each photo's best guess, and of espresso on the cat. "brightest": the largest number of each
# map and its row and column. "held": the share of the sum of a map that its 10% largest numbers hold. "fruit": on the
# drawing of fruit, the share of the square that is fruit (fruit-labels.png, resized as the photo, nearest neighbor),
# and the share of the 5% brightest pixels of the maps of balloon and orange that lie on fruit. "steps": the slider:
# for every change, the three best guesses after the 5% most salient pixels of balloon's map changed (each color one
# step of 255 per unit against its slope, clipped to 0 to 255), and after 5% random pixels changed in the same way.
# "coffee": the logit of espresso after 1, 2 and 8 steps on the 5% most salient pixels of its map, and "line", the
# change after 1 step that the straight line of *The idea* gives: the sum of each slope times the change of its number.
# "blank": the logit of Egyptian cat on the cat as it is and after the 5% most salient pixels of its map got the mean
# color of ImageNet (0 after preparing), and the change the straight line gives for that. "above": the share of each
# map that is 0.1 or more. "shared": on the cat, the share of the 5% brightest pixels of Egyptian cat's map that are
# also among the 5% brightest of tiger cat's and of espresso's map. "demo": what the demo says when the learner's
# functions are right: the probability of each photo's best guess after a change of 1 on the 5% most salient pixels
# and on 5% random pixels, and the best guess after each.
REFERENCE = {
    "logit": {'cat': 19.6482, 'coffee': 15.6166, 'fruit': 17.6986, 'espresso-on-cat': 0.0781},
    "p": {'cat': 0.3914, 'coffee': 0.8517, 'fruit': 0.7349},
    "brightest": {'cat': (0.2192, 163, 109), 'coffee': (0.3505, 90, 101), 'fruit': (0.2019, 47, 98), 'espresso-on-cat': (0.1059, 174, 134)},
    "held": {'cat': 0.2613, 'coffee': 0.3354, 'fruit': 0.2877},
    "fruit": {'area': 0.3222, 'balloon': 0.5361, 'orange': 0.7493},
    "steps": {
        0: {"salient": [('balloon', 0.7349), ('ping-pong ball', 0.1971), ('Granny Smith', 0.0277)],
            "random": [('balloon', 0.7349), ('ping-pong ball', 0.1971), ('Granny Smith', 0.0277)]},
        1: {"salient": [('ping-pong ball', 0.4396), ('balloon', 0.3961), ('Granny Smith', 0.0913)],
            "random": [('balloon', 0.63), ('ping-pong ball', 0.2734), ('Granny Smith', 0.0439)]},
        2: {"salient": [('ping-pong ball', 0.5135), ('balloon', 0.2004), ('Granny Smith', 0.177)],
            "random": [('balloon', 0.5383), ('ping-pong ball', 0.3299), ('Granny Smith', 0.0689)]},
        3: {"salient": [('ping-pong ball', 0.4237), ('Granny Smith', 0.2846), ('balloon', 0.1064)],
            "random": [('balloon', 0.4452), ('ping-pong ball', 0.375), ('Granny Smith', 0.1065)]},
        4: {"salient": [('Granny Smith', 0.3816), ('ping-pong ball', 0.2668), ('orange', 0.1752)],
            "random": [('ping-pong ball', 0.3957), ('balloon', 0.3546), ('Granny Smith', 0.1604)]},
        5: {"salient": [('Granny Smith', 0.4107), ('orange', 0.2547), ('ping-pong ball', 0.1606)],
            "random": [('ping-pong ball', 0.3697), ('balloon', 0.2818), ('Granny Smith', 0.2348)]},
        6: {"salient": [('Granny Smith', 0.3937), ('orange', 0.3229), ('ping-pong ball', 0.1011)],
            "random": [('ping-pong ball', 0.3244), ('Granny Smith', 0.315), ('balloon', 0.2144)]},
        7: {"salient": [('orange', 0.379), ('Granny Smith', 0.3591), ('croquet ball', 0.0711)],
            "random": [('Granny Smith', 0.3946), ('ping-pong ball', 0.2729), ('balloon', 0.1501)]},
        8: {"salient": [('orange', 0.4204), ('Granny Smith', 0.3212), ('croquet ball', 0.0803)],
            "random": [('Granny Smith', 0.4557), ('ping-pong ball', 0.2139), ('balloon', 0.1047)]},
        9: {"salient": [('orange', 0.4448), ('Granny Smith', 0.2904), ('croquet ball', 0.0885)],
            "random": [('Granny Smith', 0.4919), ('ping-pong ball', 0.1587), ('orange', 0.1341)]},
        10: {"salient": [('orange', 0.4618), ('Granny Smith', 0.2651), ('croquet ball', 0.0945)],
            "random": [('Granny Smith', 0.5058), ('orange', 0.1665), ('ping-pong ball', 0.117)]},
        11: {"salient": [('orange', 0.4776), ('Granny Smith', 0.243), ('croquet ball', 0.0954)],
            "random": [('Granny Smith', 0.4975), ('orange', 0.198), ('ping-pong ball', 0.0897)]},
        12: {"salient": [('orange', 0.4943), ('Granny Smith', 0.2197), ('croquet ball', 0.0932)],
            "random": [('Granny Smith', 0.4754), ('orange', 0.2263), ('croquet ball', 0.0934)]},
    },
    "coffee": {1: 11.5035, 2: 10.5358, 8: 11.2734, 'line': -7.2606},
    "blank": {'before': 19.6482, 'salient': 20.8281, 'line': 1.6557},
    "above": {'cat': 0.0027, 'coffee': 0.0082, 'fruit': 0.0012},
    "shared": {'tiger': 0.6604, 'espresso': 0.1658},
    "demo": {'cat': (0.3066, 0.3528, 'tiger cat', 'tiger cat'), 'coffee': (0.453, 0.7653, 'espresso', 'espresso'),
             'fruit': (0.3961, 0.63, 'ping-pong ball', 'balloon')},
}


def percent(share: float) -> int:
    return round(100 * share)


# --- the course's versions of the exercises ------------------------------------------------------------------------

def course_saliency(model, x, label):
    """The saliency map of photo x, of shape (1, 3, H, W), for class `label`: per pixel the largest |slope| of the
    logit of `label` over the 3 colors, a NumPy array of shape (H, W). x itself is left as it was."""
    return np.abs(course_gradient(model, x, label)).max(axis=0)


def course_most_salient(sal, share):
    """True for the share of the pixels of a map with the highest saliency."""
    return sal >= np.quantile(sal, 1 - share)


# --- the photos, the gradient and the tests of what a map promises -------------------------------------------------

def crop(photo: str) -> np.ndarray:
    """The middle 224 by 224 of a photo after step 1 of cv-prepare-photo: what ResNet-18 sees, as colors of 0 to 255."""
    return _PREP.center(_PREP.resized(str(PHOTOS[photo])))


def as_input(img) -> torch.Tensor:
    """A crop of 224 by 224 by 3, colors of 0 to 255, prepared as for ImageNet: steps 3 to 5 of cv-prepare-photo."""
    return torch.tensor(_PREP.scaled(np.asarray(img, dtype=float))).float().permute(2, 0, 1).unsqueeze(0)


def prepared(photo: str) -> torch.Tensor:
    return _PREP.prepared(str(PHOTOS[photo]))


def course_gradient(model, x, label) -> np.ndarray:
    """The slopes of the logit of `label` with respect to photo x: shape (3, H, W). x itself is left as it was."""
    x = x.clone()
    x.requires_grad = True
    model(x)[0, label].backward()
    return x.grad[0].numpy()


def random_pixels(n: int, seed: int = SEED) -> np.ndarray:
    """n random pixels of the 224 by 224, as True in a mask."""
    mask = np.zeros(SIDE * SIDE, dtype=bool)
    mask[np.random.default_rng(seed).permutation(SIDE * SIDE)[:n]] = True
    return mask.reshape(SIDE, SIDE)


def changed(img, grad: np.ndarray, mask: np.ndarray, step: int) -> np.ndarray:
    """The crop with the pixels of `mask` changed: each color `step` against its slope (down where the slope is
    positive, up where it is negative), on its scale of 0 to 255, clipped to it. Whole numbers stay whole numbers."""
    img = np.asarray(img, dtype=float)
    out = img.copy()
    out[mask] = np.clip(img[mask] - step * np.sign(grad.transpose(1, 2, 0)[mask]), 0, 255)
    return out


def blanked(x: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Photo x, prepared, with the pixels of `mask` set to 0 in every color: the mean color of ImageNet."""
    out = x.clone()
    out[0][:, torch.as_tensor(mask)] = 0
    return out


def logits_of(model, xs: list) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.cat(xs))


def probabilities(model, xs: list) -> torch.Tensor:
    return torch.softmax(logits_of(model, xs), dim=1)


def line(grad: np.ndarray, x: torch.Tensor, changed_x: torch.Tensor) -> float:
    """The change of a logit that the straight line of *The idea* gives: each slope times the change of its number,
    added up. Photos prepared, of shape (1, 3, H, W); the slopes of shape (3, H, W)."""
    return float((torch.as_tensor(grad) * (changed_x - x)[0]).sum())


def best(p: torch.Tensor, k: int = 3) -> list:
    """The k most likely classes of one row of probabilities, as (name, probability) with 4 decimals."""
    names = models.imagenet_classes()
    return [(names[int(i)], round(float(p[i]), 4)) for i in p.argsort(descending=True)[:k]]


def step_runs(model, photo: str = "fruit", label: int = BALLOON, steps=STEPS, share: float = SHARE, salient=None):
    """For every step: the probabilities after the `share` most salient pixels of the map of `label` (or the given
    mask) changed by that step against their slopes, and after as many random pixels did: two rows per step."""
    img, x = crop(photo), prepared(photo)
    grad = course_gradient(model, x, label)
    if salient is None:
        salient = course_most_salient(np.abs(grad).max(axis=0), share)
    rand = random_pixels(int(salient.sum()))
    xs = []
    for s in steps:
        xs += [as_input(changed(img, grad, salient, s)), as_input(changed(img, grad, rand, s))]
    return probabilities(model, xs)


def measure(model) -> dict:
    """Everything REFERENCE holds, measured again with the course's code."""
    out = {k: {} for k in REFERENCE}
    maps = {}
    for photo, label in TOP.items():
        x = prepared(photo)
        with torch.no_grad():
            logits = model(x)[0]
        out["logit"][photo] = round(float(logits[label]), 4)
        out["p"][photo] = round(float(torch.softmax(logits, 0)[label]), 4)
        maps[photo] = course_saliency(model, x, label)
    x = prepared("cat")
    with torch.no_grad():
        out["logit"]["espresso-on-cat"] = round(float(model(x)[0, ESPRESSO]), 4)
    maps["espresso-on-cat"] = course_saliency(model, x, ESPRESSO)
    for key, m in maps.items():
        r, c = divmod(int(m.argmax()), SIDE)
        out["brightest"][key] = (round(float(m.max()), 4), r, c)
    for photo in TOP:
        out["held"][photo] = round(held(maps[photo], 0.1), 4)
    fruit = fruit_mask()
    out["fruit"] = {"area": round(float(fruit.mean()), 4),
                    "balloon": round(on(maps["fruit"], fruit), 4),
                    "orange": round(on(course_saliency(model, prepared("fruit"), ORANGE), fruit), 4)}
    p = step_runs(model)
    out["steps"] = {s: {"salient": best(p[2 * i]), "random": best(p[2 * i + 1])} for i, s in enumerate(STEPS)}
    img, x = crop("coffee"), prepared("coffee")
    grad = course_gradient(model, x, ESPRESSO)
    top = course_most_salient(np.abs(grad).max(axis=0), SHARE)
    xs = [as_input(changed(img, grad, top, s)) for s in (1, 2, 8)]
    z = logits_of(model, xs)[:, ESPRESSO]
    out["coffee"] = {1: round(float(z[0]), 4), 2: round(float(z[1]), 4), 8: round(float(z[2]), 4), "line": round(line(grad, x, xs[0]), 4)}
    x = prepared("cat")
    grad = course_gradient(model, x, EGYPTIAN)
    top = course_most_salient(maps["cat"], SHARE)
    z = logits_of(model, [x, blanked(x, top)])[:, EGYPTIAN]
    out["blank"] = {"before": round(float(z[0]), 4), "salient": round(float(z[1]), 4), "line": round(line(grad, x, blanked(x, top)), 4)}
    out["above"] = {photo: round(float((maps[photo] >= 0.1).mean()), 4) for photo in TOP}
    tiger = course_saliency(model, x, TIGER)
    out["shared"] = {"tiger": round(shared(maps["cat"], tiger), 4), "espresso": round(shared(maps["cat"], maps["espresso-on-cat"]), 4)}
    for photo, label in TOP.items():
        p = step_runs(model, photo, label, steps=[1])
        out["demo"][photo] = (round(float(p[0, label]), 4), round(float(p[1, label]), 4), best(p[0], 1)[0][0], best(p[1], 1)[0][0])
    return out


def held(sal: np.ndarray, share: float) -> float:
    """The share of the sum of a map that its `share` largest numbers hold."""
    flat = np.sort(sal.ravel())[::-1]
    return float(flat[:round(share * flat.size)].sum() / flat.sum())


def fruit_mask() -> np.ndarray:
    """Where the drawing of fruit shows fruit, in the square ResNet-18 sees: its labels, resized as the photo with the
    nearest label, and the same middle square."""
    from PIL import Image
    lab = Image.open(FRUIT_LABELS)
    return _PREP.center(np.array(lab.resize(_PREP.size_after(*lab.size, _PREP.SHORT), Image.NEAREST))) > 0


def on(sal: np.ndarray, where: np.ndarray, share: float = SHARE) -> float:
    """The share of the `share` brightest pixels of a map that lie where `where` is True."""
    top = course_most_salient(sal, share)
    return float((top & where).sum() / top.sum())


def shared(a: np.ndarray, b: np.ndarray, share: float = SHARE) -> float:
    """The share of the `share` brightest pixels of map a that are also among those of map b."""
    ta, tb = course_most_salient(a, share), course_most_salient(b, share)
    return float((ta & tb).sum() / ta.sum())


# --- the pictures and the sentences under them ---------------------------------------------------------------------

_INSIDE = _lesson_checks("cv-inside-cnn")   # how a map is drawn there: _slim and _bare


def _group(pairs: list, top: float, title: str, sep: str) -> list:
    """One group of the slider's picture: a title, then a bar per class, as long as its probability on one scale
    for every group and every stop (a probability of 1 is BW long), with the probability at its end. The bars are what the
    model puts out, so they have the color of a prediction; the best guess is full, and balloon is bold."""
    LW, BW, ROW = 122, 160, 22
    parts = [f'<text x="0" y="{top + 15}" style="font-size:14px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>']
    for j, (name, p) in enumerate(pairs):
        y = top + 24 + j * ROW
        w = max(BW * p, 2)
        parts.append(f'<text x="{LW - 8}" y="{y + 13}" style="font-size:13px;fill:var(--ea-ink);text-anchor:end;'
                     f'font-weight:{700 if name == "balloon" else 400}">{html.escape(_TRY._short(name))}</text>'
                     f'<rect x="{LW}" y="{y + 2}" width="{w:.1f}" height="14" rx="3" style="fill:var(--ea-pred);opacity:{1 if j == 0 else 0.45}"/>'
                     f'<text x="{LW + w + 6:.1f}" y="{y + 14}" style="font-size:12.5px;fill:var(--ea-pred-text)">{html.escape(_TRY.shown(p, sep))}</text>')
    return parts


def step_frames(labels: dict, lang: str) -> dict:
    """For the slider in *The idea*: for every change, the three best guesses for the drawing of fruit after the 5%
    most salient pixels of balloon's map changed, and after 5% random pixels did, from REFERENCE. Drawn 330 wide, for
    a phone. `labels` gives "salient" and "random" ({step}), and "alt" ({step}, {salient} and {random})."""
    sep = "," if lang == "nl" else "."
    W, GAP = 330, 106
    frames = {}
    for step in STEPS:
        row = REFERENCE["steps"][step]
        parts = _group(row["salient"], 0, labels["salient"].format(step=step), sep) + _group(row["random"], GAP, labels["random"].format(step=step), sep)
        said = {k: ", ".join(f"{n} {_TRY.shown(p, sep)}" for n, p in row[k]) for k in ("salient", "random")}
        alt = labels["alt"].format(step=step, salient=said["salient"], random=said["random"])
        H = GAP + 24 + 3 * 22 + 4
        frames[step] = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(alt)}" style="width:100%;max-width:{W}px;height:auto;font-family:var(--ea-font-ui)">'
                        f'<title>{html.escape(alt)}</title>' + "".join(parts) + "</svg>")
    return frames


def _guess(pairs: list, lang: str) -> str:
    """The best guess of three, and balloon's probability when balloon is not the best guess."""
    sep = "," if lang == "nl" else "."
    name, p = pairs[0]
    if name == "balloon":
        return {"en": f"balloon stays the best guess, at {_TRY.shown(p, sep)}",
                "nl": f"blijft balloon de beste gok, met {_TRY.shown(p, sep)}"}[lang]
    ball = dict(pairs).get("balloon")
    rest = ({"en": f"; balloon {_TRY.shown(ball, sep)}", "nl": f"; balloon {_TRY.shown(ball, sep)}"}[lang] if ball is not None
            else {"en": "; balloon is no longer among the three", "nl": "; balloon staat niet meer bij de drie"}[lang])
    return {"en": f"{name} is the best guess, at {_TRY.shown(p, sep)}{rest}",
            "nl": f"is {name} de beste gok, met {_TRY.shown(p, sep)}{rest}"}[lang]


def step_sentence(step: int, lang: str) -> str:
    row = REFERENCE["steps"][step]
    if step == 0:
        return {"en": "Nothing changed yet: balloon is the best guess, at 0.73.",
                "nl": "Er is nog niets veranderd: balloon is de beste gok, met 0,73."}[lang]
    if lang == "nl":
        return (f"Elke kleur {step} omlaag of omhoog. Op de 5% pixels met de hoogste saliency {_guess(row['salient'], lang)}. "
                f"Op 5% willekeurige pixels {_guess(row['random'], lang)}.")
    return (f"Each color {step} down or up. On the 5% most salient pixels, {_guess(row['salient'], lang)}. "
            f"On 5% random pixels, {_guess(row['random'], lang)}.")


def first_lost(kind: str) -> int:
    """The smallest change at which balloon is no longer the best guess, on the most salient or on random pixels."""
    return min(s for s in STEPS if REFERENCE["steps"][s][kind][0][0] != "balloon")


def _draw_map(ax, sal):
    """A map as a gray image: black 0, white from its quantile WHITE on, the 1% largest numbers."""
    ax.imshow(sal, cmap="gray", vmin=0, vmax=float(np.quantile(sal, WHITE)), interpolation="none")
    _INSIDE._bare(ax)


def maps_picture(maps: dict, labels: dict) -> str:
    """The three photos as ResNet-18 sees them, and under each its saliency map for its best guess. `labels` gives
    "names" (the name of each photo), "map" ({label}: the class) and "alt"."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    names = models.imagenet_classes()
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.75), gridspec_kw={"wspace": 0.06, "hspace": 0.16})
    for col, photo in enumerate(TOP):
        axes[0, col].imshow(crop(photo), interpolation="none")
        _INSIDE._bare(axes[0, col])
        axes[0, col].set_title(labels["names"][photo], fontsize=10)
        _draw_map(axes[1, col], maps[photo])
        axes[1, col].set_title(labels["map"].format(label=names[TOP[photo]]), fontsize=10)
    for ax in axes.ravel():
        for side in ax.spines.values():
            side.set_color("none")
    out = _INSIDE._slim(plots.render(fig, alt=labels["alt"]))
    plt.close(fig)
    return out


def faded(img, mask: np.ndarray) -> np.ndarray:
    """The crop with the pixels of `mask` in their own colors and the others faded to a quarter, on white."""
    img = np.asarray(img, dtype=float) / 255
    out = 0.25 * img + 0.75
    out[mask] = img[mask]
    return out


def demo_picture(result: dict, labels: dict) -> str:
    """The demo: the photo with the pixels of the learner's most_salient in color and the rest faded, and the
    learner's map. `labels` gives "mask", "map" and "alt"."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.7), gridspec_kw={"wspace": 0.08})
    left.imshow(faded(result["photo"], result["mask"]), interpolation="none")
    _INSIDE._bare(left)
    left.set_title(labels["mask"], fontsize=10)
    _draw_map(right, result["map"])
    right.set_title(labels["map"], fontsize=10)
    for ax in (left, right):
        for side in ax.spines.values():
            side.set_color("none")
    out = _INSIDE._slim(plots.render(fig, alt=labels["alt"]))
    plt.close(fig)
    return out


def demo_sentence(result: dict, photo: str, lang: str) -> str:
    """What the demo measured, with the learner's map and mask."""
    sep = "," if lang == "nl" else "."
    name = models.imagenet_classes()[TOP[photo]]
    before, after, rand = (_TRY.share(result[k], sep) for k in ("before", "salient", "random"))
    if lang == "nl":
        first, other = ((f"blijft {name} de beste gok" if result[k] == name else f"wordt {result[k]} de beste gok") for k in ("best", "best_random"))
        return (f"Je map voor {name}, met in kleur de 5% pixels die je `most_salient` kiest. Verandert elke kleur van die "
                f"pixels 1 tegen zijn helling in, dan gaat {name} van {before} naar {after}, en {first}. Bij 5% "
                f"willekeurige pixels, op dezelfde manier veranderd: {rand}, en {other}.")
    first, other = ((f"{name} stays the best guess" if result[k] == name else f"{result[k]} becomes the best guess") for k in ("best", "best_random"))
    return (f"Your map for {name}, with the 5% of pixels your `most_salient` picks in color. Change each color of those "
            f"pixels by 1 against its slope, and {name} goes from {before} to {after}, and {first}. On 5% random "
            f"pixels, changed in the same way: {rand}, and {other}.")


# --- the exercises ---------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "brighter" and "alike" are the choice questions in the lesson.
REQUIRED = ["brighter", "saliency", "most-salient", "alike"]

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
    """The same value, with floats within 1 in 10,000 of each other, also when they are small: the slopes of a
    ResNet-18 with random weights are thousandths."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool) or math.isnan(got):
            return False
        return abs(got - want) <= 1e-4 * abs(want) + 1e-9
    return got == want


_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

# --- exercise 2: the saliency map ----------------------------------------------------------------------------------

_SAL_HEAD = "import numpy as np\nimport torch\n\n\ndef saliency(model, x, label):\n"
_SAL_START = (_SAL_HEAD + "    # model: a classifier in eval mode; x: one photo, a tensor of shape (1, 3, H, W)\n"
              "    # give back the saliency map of x for the class label, of shape (H, W):\n"
              "    # per pixel, the largest |slope| of the logit of label over the 3 colors\n    ...\n")
_SAL_BODY = ("    x.requires_grad = True\n    x.grad = None\n    score = model(x)[0, label]\n    score.backward()\n"
             "    g = x.grad[0].numpy()\n    return np.abs(g).max(axis=0)\n")


def _sal(body: str = _SAL_BODY) -> str:
    return _SAL_HEAD + body


def _sal_end(end: str, score: str = "model(x)[0, label]", reset: bool = True) -> str:
    """The solution with another score, or another last step after `g = x.grad`."""
    return _sal("    x.requires_grad = True\n" + ("    x.grad = None\n" if reset else "") + f"    score = {score}\n    score.backward()\n" + end)


_SAL = _sal()
_SAL_MODEL = ("from everything_ai.models import ResNet18\n\ntorch.manual_seed(0)\n"
              "model = ResNet18().eval()     # random weights, not trained: the check needs no download\n"
              "model.requires_grad_(False)   # as in the course's ResNet-18: the weights keep no slopes\n")
_SAL_SCENARIOS = [
    # a photo of 32 by 32: the shape of the map, its sum, its largest number and where that is
    _SAL_MODEL + "x = torch.randn(1, 3, 32, 32)\nsal = saliency(model, x, 3)\n"
    "[list(sal.shape), float(sal.sum()), float(sal.max()), int(sal.argmax())]",
    # a photo of another size, 40 by 48, and another class
    _SAL_MODEL + "x = torch.randn(1, 3, 40, 48)\nsal = saliency(model, x, 7)\n[list(sal.shape), float(sal.sum()), float(sal[10, 20])]",
    # one photo, two classes, one after the other
    _SAL_MODEL + "x = torch.randn(1, 3, 32, 32)\na = saliency(model, x, 3)\nb = saliency(model, x, 7)\n[float(a.sum()), float(b.sum())]",
]
_SAL_WRONG = {
    "no-abs": _sal_end("    g = x.grad[0].numpy()\n    return g.max(axis=0)\n"),
    "sum-colors": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).sum(axis=0)\n"),
    "mean-colors": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).mean(axis=0)\n"),
    "probability": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).max(axis=0)\n", score="torch.softmax(model(x), dim=1)[0, label]"),
    "top-class": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).max(axis=0)\n", score="model(x).max()"),
    "adds-up": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).max(axis=0)\n", reset=False),
    "gradient": _sal_end("    return x.grad[0]\n"),
    "batch": _sal_end("    g = x.grad.numpy()\n    return np.abs(g).max(axis=0)\n"),
    "keeps-batch": _sal_end("    g = x.grad.numpy()\n    return np.abs(g).max(axis=1)\n"),
    "rows": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).max(axis=1)\n"),
    "colors-last": _sal_end("    g = x.grad[0].numpy()\n    return np.abs(g).max(axis=2)\n"),
}

# --- exercise 3: the most salient pixels ---------------------------------------------------------------------------

_MS_HEAD = "import numpy as np\n\n\ndef most_salient(sal, share):\n"
_MS_START = (_MS_HEAD + "    # sal: a saliency map, a NumPy array of shape (H, W); share: a number from 0 to 1\n"
             "    # give back an array of True and False of the same shape: True for the share of\n"
             "    # the pixels with the highest saliency, such as the 5% for share 0.05\n    ...\n")
_MS_BODY = "    return sal >= np.quantile(sal, 1 - share)\n"


def _ms(body: str = _MS_BODY) -> str:
    return _MS_HEAD + body


_MS = _ms()
_MS_SCENARIOS = [
    # a map of 4 by 5 with the numbers 0 to 1 in order: a quarter of 20 pixels is the 5 brightest, the bottom row
    "sal = np.arange(20).reshape(4, 5) / 19\nmask = most_salient(sal, 0.25)\nmask.tolist()",
    # a random map of 50 by 40, of thousandths as a saliency map is: 5% of its 2,000 pixels, and the sum of their numbers
    "sal = np.random.default_rng(0).random((50, 40)) / 1000\nmask = most_salient(sal, 0.05)\n[int(mask.sum()), float(sal[mask].sum())]",
    # half of a random map of 6 by 4, of thousandths, and its top row
    "sal = np.random.default_rng(1).random((6, 4)) / 1000\nmask = most_salient(sal, 0.5)\n[int(mask.sum()), mask[0].tolist()]",
]
_MS_WRONG = {
    "flipped": _ms("    return sal >= np.quantile(sal, share)\n"),
    "lowest": _ms("    return sal <= np.quantile(sal, share)\n"),
    "threshold": _ms("    return np.quantile(sal, 1 - share)\n"),
    "value": _ms("    return sal >= 1 - share\n"),
    "percentile": _ms("    return sal >= np.percentile(sal, 1 - share)\n"),
}


EXERCISES = {
    "saliency": ScenarioBox(
        name="saliency",
        start=_SAL_START,
        solution=_SAL,
        scenarios=_SAL_SCENARIOS,
        compare=_close,
        digits=5,
        save_as=("cnn", "saliency"),
        first=[_DOTS],
        mistakes=[
            Mistake("not-recorded", error=lambda e: isinstance(e, RuntimeError) and "does not require grad" in str(e),
                    en="`backward` finds nothing to go back to: PyTorch recorded nothing on the way from the photo to the score. The weights of the model keep no slopes, so the photo must: set `x.requires_grad = True` before you take it through the model. And not inside `with torch.no_grad():`, where PyTorch records nothing.",
                    nl="`backward` vindt niets om naar terug te gaan: PyTorch heeft niets opgenomen op de weg van de foto naar de score. De weights van het model houden geen hellingen bij, dus moet de foto dat doen: zet `x.requires_grad = True` voordat je hem door het model haalt. En niet binnen `with torch.no_grad():`, waar PyTorch niets opneemt."),
            Mistake("one-number", error=lambda e: isinstance(e, RuntimeError) and "only for scalar outputs" in str(e),
                    en="`backward` starts from one number, and you call it on more: `model(x)` gives a logit for every class, shape (1, classes). Take the logit of `label` for photo 0: `model(x)[0, label]`.",
                    nl="`backward` begint bij één getal, en je roept hem aan op meer: `model(x)` geeft een logit voor elke class, shape (1, classes). Neem de logit van `label` voor foto 0: `model(x)[0, label]`."),
            Mistake("batch-index", error=lambda e: isinstance(e, IndexError) and "dimension 0 with size 1" in str(e) and _raised_in(e, "saliency"),
                    en="`model(x)` has shape (1, classes): one row, for the one photo of the batch. `model(x)[label]` asks for row `label`, which is not there. Take row 0 and then the class: `model(x)[0, label]`.",
                    nl="`model(x)` heeft shape (1, classes): één rij, voor de ene foto van de batch. `model(x)[label]` vraagt rij `label`, en die is er niet. Neem rij 0 en dan de class: `model(x)[0, label]`."),
            Mistake("numpy-recorded", error=lambda e: isinstance(e, RuntimeError) and "requires grad" in str(e) and "numpy" in str(e),
                    en="`.numpy()` does not work on a tensor that PyTorch records, such as `x` itself after `x.requires_grad = True`. The slopes are in `x.grad`, which PyTorch does not record: `x.grad[0].numpy()`.",
                    nl="`.numpy()` werkt niet op een tensor die PyTorch opneemt, zoals `x` zelf na `x.requires_grad = True`. De hellingen staan in `x.grad`, en die neemt PyTorch niet op: `x.grad[0].numpy()`."),
            Mistake("zero-none", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute 'zero_'" in str(e),
                    en="`x.grad` is `None` until the first `backward`, so `x.grad.zero_()` has nothing to clear and stops. Clear it the other way, which works whether there is a slope or not: `x.grad = None`, before `backward`.",
                    nl="`x.grad` is `None` tot de eerste `backward`, dus heeft `x.grad.zero_()` niets om leeg te maken, en stopt. Maak hem op de andere manier leeg, die werkt of er nu een helling is of niet: `x.grad = None`, vóór `backward`."),
            Mistake("no-backward", error=lambda e: isinstance(e, (TypeError, AttributeError)) and "'NoneType'" in str(e) and _raised_in(e, "saliency"),
                    en="`x.grad` is still `None`: there is no slope yet. `score.backward()` fills it; call it after you take the logit, and before you read `x.grad`.",
                    nl="`x.grad` is nog `None`: er is nog geen helling. `score.backward()` vult hem; roep hem aan nadat je de logit neemt, en voordat je `x.grad` leest."),
            Mistake("max-pair", error=lambda e: isinstance(e, AttributeError) and "return_types" in str(e),
                    en="`.max(dim=0)` of a tensor gives two things, the largest numbers and where they are, and the check gets that pair. `np.abs` of a tensor stays a tensor, so `.max(axis=0)` on it does the same. Make the slopes a NumPy array first, and take the largest along axis 0: `np.abs(x.grad[0].numpy()).max(axis=0)`.",
                    nl="`.max(dim=0)` van een tensor geeft twee dingen, de grootste getallen en waar ze staan, en de check krijgt dat paar. `np.abs` van een tensor blijft een tensor, dus doet `.max(axis=0)` erop hetzelfde. Maak van de hellingen eerst een NumPy-array, en neem de grootste langs axis 0: `np.abs(x.grad[0].numpy()).max(axis=0)`."),
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e) and not _raised_in(e, "saliency"),
                    en="The check asks for the `.shape` of what your function gives back, and it gives back nothing, `None`: end with `return` and the map.",
                    nl="De check vraagt de `.shape` van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de map."),
            Mistake("adds-up", wrong=_SAL_WRONG["adds-up"],
                    en=_S_EN + ": the first map is right, the second is not. The second call adds its slopes to those of the first, in the same `x.grad`, as `.grad` did in [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en): its map is of class 3 and class 7 together. Set `x.grad = None` before `backward`.",
                    nl=_S_NL + ": de eerste map klopt, de tweede niet. De tweede aanroep telt zijn hellingen op bij die van de eerste, in dezelfde `x.grad`, zoals `.grad` deed in [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl): zijn map is van class 3 en class 7 samen. Zet `x.grad = None` vóór `backward`."),
            Mistake("no-abs", wrong=_SAL_WRONG["no-abs"],
                    en=_S_EN + ": you take the largest slope with its sign, so a pixel whose slopes are all negative gets a negative number, however much it changes the score. Take the size first, `np.abs`, then the largest.",
                    nl=_S_NL + ": je neemt de grootste helling met zijn teken, dus krijgt een pixel waarvan alle hellingen negatief zijn een negatief getal, hoeveel hij de score ook verandert. Neem eerst de grootte, `np.abs`, dan de grootste."),
            Mistake("sum-colors", wrong=_SAL_WRONG["sum-colors"],
                    en=_S_EN + ": you add the sizes of the three slopes of a pixel. That is a map too, but not this lesson's: take the largest of the three, `.max(axis=0)`, as Simonyan and colleagues did.",
                    nl=_S_NL + ": je telt de groottes van de drie hellingen van een pixel op. Dat is ook een map, maar niet die van deze les: neem de grootste van de drie, `.max(axis=0)`, zoals Simonyan en zijn collega's deden."),
            Mistake("mean-colors", wrong=_SAL_WRONG["mean-colors"],
                    en=_S_EN + ": you take the mean of the sizes of the three slopes of a pixel. Take the largest of the three: `.max(axis=0)`.",
                    nl=_S_NL + ": je neemt het gemiddelde van de groottes van de drie hellingen van een pixel. Neem de grootste van de drie: `.max(axis=0)`."),
            Mistake("probability", wrong=_SAL_WRONG["probability"],
                    en=_S_EN + ": you take the slopes of the probability. A probability can also rise because other classes fall, so its slopes mix in all of them; and when one class has almost everything, they are close to 0. Take the logit itself: `model(x)[0, label]`.",
                    nl=_S_NL + ": je neemt de hellingen van de kans. Een kans kan ook stijgen doordat andere classes dalen, dus mengen zijn hellingen ze er allemaal in; en als één class bijna alles heeft, liggen ze dicht bij 0. Neem de logit zelf: `model(x)[0, label]`."),
            Mistake("top-class", wrong=_SAL_WRONG["top-class"],
                    en=_S_EN + ": `model(x).max()` is the largest logit, the class the model chose, whatever `label` is. Take the logit of `label`: `model(x)[0, label]`.",
                    nl=_S_NL + ": `model(x).max()` is de grootste logit, de class die het model koos, wat `label` ook is. Neem de logit van `label`: `model(x)[0, label]`."),
            Mistake("gradient", wrong=_SAL_WRONG["gradient"],
                    en=_S_EN + ": you give back the slopes themselves, one per color, shape (3, H, W). A map has one number per pixel: the largest size of its three slopes, `np.abs(...).max(axis=0)`.",
                    nl=_S_NL + ": je geeft de hellingen zelf terug, één per kleur, shape (3, H, W). Een map heeft één getal per pixel: de grootste grootte van zijn drie hellingen, `np.abs(...).max(axis=0)`."),
            Mistake("batch", wrong=_SAL_WRONG["batch"],
                    en=_S_EN + ": `x.grad` has the batch first, shape (1, 3, H, W), so axis 0 is the batch, and your map is the sizes of the slopes per color, shape (3, H, W). Take photo 0 first: `x.grad[0]`, then the largest along axis 0, the colors.",
                    nl=_S_NL + ": `x.grad` heeft de batch vooraan, shape (1, 3, H, W), dus is axis 0 de batch, en is je map de groottes van de hellingen per kleur, shape (3, H, W). Neem eerst foto 0: `x.grad[0]`, dan de grootste langs axis 0, de kleuren."),
            Mistake("keeps-batch", wrong=_SAL_WRONG["keeps-batch"],
                    en=_S_EN + ": the numbers are right, but your map keeps the batch: shape (1, H, W). Take photo 0 first, `x.grad[0]`, and then the largest along axis 0.",
                    nl=_S_NL + ": de getallen kloppen, maar je map houdt de batch: shape (1, H, W). Neem eerst foto 0, `x.grad[0]`, en dan de grootste langs axis 0."),
            Mistake("rows", wrong=_SAL_WRONG["rows"],
                    en=_S_EN + ": `x.grad[0]` has shape (3, H, W), the colors first, and you take the largest along axis 1, the rows. Take it along axis 0: `.max(axis=0)`.",
                    nl=_S_NL + ": `x.grad[0]` heeft shape (3, H, W), de kleuren vooraan, en je neemt de grootste langs axis 1, de rijen. Neem hem langs axis 0: `.max(axis=0)`."),
            Mistake("colors-last", wrong=_SAL_WRONG["colors-last"],
                    en=_S_EN + ": in PyTorch the colors come first, `x.grad[0]` has shape (3, H, W), not last as in a NumPy image of shape (H, W, 3). Along axis 2 you take the largest of each row. Take it along axis 0, the colors: `.max(axis=0)`.",
                    nl=_S_NL + ": in PyTorch staan de kleuren vooraan, `x.grad[0]` heeft shape (3, H, W), niet achteraan zoals in een NumPy-beeld met shape (H, W, 3). Langs axis 2 neem je de grootste van elke rij. Neem hem langs axis 0, de kleuren: `.max(axis=0)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: record the photo, take the logit of `label`, go back from it, and make a map of the slopes.",
                      "nl": "Vervang de `...` door je code: zet het opnemen aan voor de foto, neem de logit van `label`, ga ervandaan terug, en maak een map van de hellingen."},
            "wrong": {"en": _S_EN + ": the shape of the map, its sum, its largest number and where that is. Is it the logit of `label` you go back from, and is each number of the map the largest of the sizes of a pixel's three slopes?",
                      "nl": _S_NL + ": de shape van de map, zijn som, zijn grootste getal en waar dat staat. Ga je terug van de logit van `label`, en is elk getal van de map de grootste van de groottes van de drie hellingen van een pixel?"},
            "almost": {"en": "Right for the first photo. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for a photo of any size, and for two maps of one photo, one after the other?",
                       "nl": "Goed voor de eerste foto. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor een foto van elke maat, en voor twee maps van één foto, na elkaar?"},
            "correct": {"en": "Your `saliency` is right: one backward pass gives a slope for every pixel and color, and the map keeps the largest size per pixel. It is now in your project, as `{saved}`.",
                        "nl": "Je `saliency` klopt: één backward pass geeft een helling voor elke pixel en kleur, en de map houdt per pixel de grootste grootte. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `saliency` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `saliency` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "most-salient": ScenarioBox(
        name="most-salient",
        start=_MS_START,
        solution=_MS,
        scenarios=_MS_SCENARIOS,
        compare=_close,
        digits=7,
        save_as=("cnn", "salient"),
        first=[_DOTS],
        mistakes=[
            Mistake("percent", error=lambda e: isinstance(e, ValueError) and "Quantiles must be in the range" in str(e),
                    en="`np.quantile` takes a share from 0 to 1, not a percentage: the 0.95 quantile is the number that 95% of the map is below. With a share of 0.05, that is `1 - share`.",
                    nl="`np.quantile` wil een deel van 0 tot 1, geen percentage: het 0,95-kwantiel is het getal waar 95% van de map onder ligt. Met een deel van 0,05 is dat `1 - share`."),
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for what your function gives back, and it gives back nothing, `None`: end with `return` and the array of True and False.",
                    nl="De check vraagt wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de array met True en False."),
            Mistake("flipped", wrong=_MS_WRONG["flipped"],
                    en=_S_EN + ": `np.quantile(sal, share)` is the number that a share `share` of the map is below, so `>=` keeps the other 75%, and for 0.05 the other 95%. The brightest share starts at `np.quantile(sal, 1 - share)`.",
                    nl=_S_NL + ": `np.quantile(sal, share)` is het getal waar een deel `share` van de map onder ligt, dus houdt `>=` de andere 75%, en bij 0,05 de andere 95%. Het lichtste deel begint bij `np.quantile(sal, 1 - share)`."),
            Mistake("lowest", wrong=_MS_WRONG["lowest"],
                    en=_S_EN + ": that is the share of the pixels with the lowest saliency, the darkest of the map. The brightest share is at or above `np.quantile(sal, 1 - share)`.",
                    nl=_S_NL + ": dat is het deel van de pixels met de laagste saliency, de donkerste van de map. Het lichtste deel ligt op of boven `np.quantile(sal, 1 - share)`."),
            Mistake("threshold", wrong=_MS_WRONG["threshold"],
                    en=_S_EN + ": you give back the number where the brightest share starts, one number. Compare the map with it: `sal >= ...` gives True or False for every pixel.",
                    nl=_S_NL + ": je geeft het getal terug waar het lichtste deel begint, één getal. Vergelijk de map ermee: `sal >= ...` geeft True of False voor elke pixel."),
            Mistake("value", wrong=_MS_WRONG["value"],
                    en=_S_EN + ": `sal >= 1 - share` compares the numbers of the map with `1 - share` itself, and a map can have any numbers: this one, of thousandths, has none that large. The share is about how many pixels: find the number that a share `1 - share` of the map is below, with `np.quantile`.",
                    nl=_S_NL + ": `sal >= 1 - share` vergelijkt de getallen van de map met `1 - share` zelf, en een map kan elke getallen hebben: deze, van duizendsten, heeft er geen die zo groot zijn. Het deel gaat over hoeveel pixels: zoek het getal waar een deel `1 - share` van de map onder ligt, met `np.quantile`."),
            Mistake("percentile", wrong=_MS_WRONG["percentile"],
                    en=_S_EN + ": `np.percentile` takes a percentage, from 0 to 100, so `1 - share` is almost its lowest number, and almost every pixel passes. Use `np.quantile(sal, 1 - share)`, or `np.percentile(sal, 100 * (1 - share))`.",
                    nl=_S_NL + ": `np.percentile` wil een percentage, van 0 tot 100, dus is `1 - share` bijna zijn laagste getal, en komt bijna elke pixel erdoor. Gebruik `np.quantile(sal, 1 - share)`, of `np.percentile(sal, 100 * (1 - share))`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: find the number where the brightest share of the map starts, and compare the map with it.",
                      "nl": "Vervang de `...` door je code: zoek het getal waar het lichtste deel van de map begint, en vergelijk de map ermee."},
            "wrong": {"en": _S_EN + ". Is it True for the brightest pixels, and for as many as the share says?",
                      "nl": _S_NL + ". Is het True voor de lichtste pixels, en voor zoveel als het deel zegt?"},
            "almost": {"en": "Right for the first map. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any map and any share?",
                       "nl": "Goed voor de eerste map. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elke map en elk deel?"},
            "correct": {"en": "Your `most_salient` is right: True for the share of the pixels with the highest saliency, such as the 5% that the slider changed. It is now in your project, as `{saved}`.",
                        "nl": "Je `most_salient` klopt: True voor het deel van de pixels met de hoogste saliency, zoals de 5% die de schuifregelaar veranderde. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `most_salient` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `most_salient` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("saliency", _SAL_START, "empty", None),
    ("saliency", _SAL, "correct", None),
    # in torch to the end, with amax: a tensor works as well as a NumPy array
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    return x.grad[0].abs().amax(dim=0)\n"), "correct", None),
    # a copy of the photo each time: no old slope to clear
    ("saliency", _sal("    x = x.clone()\n    x.requires_grad = True\n    model(x)[0][label].backward()\n    return np.abs(x.grad.numpy()[0]).max(axis=0)\n"), "correct", None),
    ("saliency", _sal("    x = x.detach()\n    x.requires_grad = True\n    score = model(x)[:, label].sum()\n    score.backward()\n"
                      "    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "correct", None),
    # with torch.autograd.grad: the slopes without .grad at all
    ("saliency", _sal("    x = x.clone()\n    x.requires_grad = True\n    (g,) = torch.autograd.grad(model(x)[0, label], x)\n    return g[0].abs().max(dim=0).values\n"), "correct", None),
    ("saliency", _sal("    score = model(x)[0, label]\n    score.backward()\n    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "not-recorded"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    with torch.no_grad():\n        score = model(x)[0, label]\n    score.backward()\n"
                      "    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "not-recorded"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x).backward()\n    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "one-number"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[label].backward()\n    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "batch-index"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    score = model(x)[0, label]\n    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "no-backward"),
    # clearing the slopes as in a training step, before there are any
    ("saliency", _sal("    x.requires_grad = True\n    x.grad.zero_()\n    model(x)[0, label].backward()\n    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "zero-none"),
    # a guarded zero_ works
    ("saliency", _sal("    x.requires_grad = True\n    if x.grad is not None:\n        x.grad.zero_()\n    model(x)[0, label].backward()\n    return np.abs(x.grad[0].numpy()).max(axis=0)\n"), "correct", None),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    return np.abs(x.numpy()).max(axis=0)\n"), "error", "numpy-recorded"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    return x.grad[0].abs().max(dim=0)\n"), "error", "max-pair"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    return np.abs(x.grad[0]).max(axis=0)\n"), "error", "max-pair"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    return np.abs(x.grad[0].numpy()).max(axis=-1)\n"), "wrong", "colors-last"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    sal = np.abs(x.grad[0].numpy()).max(axis=0)\n"), "error", "no-return"),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    ...\n"), "wrong", "dots-left"),
    # right for one photo of 32 by 32 only
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    sal = np.abs(x.grad[0].numpy()).max(axis=0)\n"
                      "    if sal.shape[1] != 32:\n        return sal * 2\n    return sal\n"), "almost", None),
    ("saliency", _sal("    x.requires_grad = True\n    x.grad = None\n    model(x)[0, label].backward()\n    return np.abs(x.grad[0].numpy()).max(axis=0) * 2\n"), "wrong", None),
] + [("saliency", code, "wrong", key) for key, code in _SAL_WRONG.items()] + [
    ("most-salient", _MS_START, "empty", None),
    ("most-salient", _MS, "correct", None),
    ("most-salient", _ms("    return sal > np.quantile(sal, 1 - share)\n"), "correct", None),
    ("most-salient", _ms("    line = np.quantile(sal, 1 - share)\n    keep = sal >= line\n    return keep\n"), "correct", None),
    ("most-salient", _ms("    return sal >= np.percentile(sal, 100 * (1 - share))\n"), "correct", None),
    # the k brightest by sorting
    ("most-salient", _ms("    k = round(share * sal.size)\n    line = np.sort(sal.ravel())[::-1][k - 1]\n    return sal >= line\n"), "correct", None),
    ("most-salient", _ms("    return sal >= np.quantile(sal, 100 - share)\n"), "error", "percent"),
    ("most-salient", _ms("    keep = sal >= np.quantile(sal, 1 - share)\n"), "error", "no-return"),
    ("most-salient", _ms("    return ...\n"), "wrong", "dots-left"),
    # right for a quarter only
    ("most-salient", _ms("    if share == 0.25:\n        return sal >= np.quantile(sal, 0.75)\n    return sal >= np.quantile(sal, 0.5)\n"), "almost", None),
    ("most-salient", _ms("    return sal >= np.median(sal)\n"), "wrong", None),
] + [("most-salient", code, "wrong", key) for key, code in _MS_WRONG.items()]


# --- the demo: the learner's functions on ResNet-18 ------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def demo(photo: str):
    """The demo's result for a photo, worked out anew each time, in a few seconds: None when saliency.py or salient.py
    is not in the project yet; "saliency" or "salient" when the learner's function stops with an error or gives
    something of another shape; else a dict with the photo as ResNet-18 sees it ("photo"), the learner's map ("map")
    and their 5% most salient pixels ("mask"), and the probability of the photo's best guess as it is ("before"),
    after a change of 1 on those pixels ("salient", with the best guess then, "best") and on as many random pixels
    ("random", and "best_random")."""
    if not (_saved("saliency") and _saved("salient")):
        return None
    run = run_code("from cnn.saliency import saliency")
    if run.error is not None:
        return "saliency"
    saliency = run.names["saliency"]
    run = run_code("from cnn.salient import most_salient")
    if run.error is not None:
        return "salient"
    most_salient = run.names["most_salient"]
    model = _TRY._resnet18()
    label = TOP[photo]
    try:
        sal = with_time_limit(lambda: np.asarray(torch.as_tensor(saliency(model, prepared(photo), label)).detach(), dtype=float), 30)
        if sal.shape != (SIDE, SIDE) or not np.isfinite(sal).all() or not sal.max() > 0:
            return "saliency"
    except Exception:
        return "saliency"
    finally:   # the learner's function may have changed the course's model: put it back as the course uses it
        model.eval()
        model.requires_grad_(False)
    try:
        mask = with_time_limit(lambda: np.asarray(torch.as_tensor(np.asarray(most_salient(sal.copy(), SHARE)))), 10)
        if mask.shape != (SIDE, SIDE) or not np.isin(mask, (0, 1)).all():
            return "salient"
        mask = mask.astype(bool)
        if not 0 < mask.sum() < mask.size:
            return "salient"
    except Exception:
        return "salient"
    p = step_runs(model, photo, label, steps=[0, 1], salient=mask)
    return {"photo": crop(photo), "map": sal, "mask": mask, "before": float(p[0, label]), "salient": float(p[2, label]),
            "random": float(p[3, label]), "best": best(p[2], 1)[0][0], "best_random": best(p[3], 1)[0][0]}
