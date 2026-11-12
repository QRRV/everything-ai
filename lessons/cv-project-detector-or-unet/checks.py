# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the final project lesson cv-project-detector-or-unet: harder, which makes a scene of the course harder to
read with fog, light from one side and noise, and shares, which gives for every group of the shapes the share a model
found. Both are saved in the learner's project (work/projects/cnn/harder.py and shares.py) when right. The demo trains
the network the learner chooses, their U-Net of cv-unet or their detector of cv-one-stage-detector, on the course's
2,000 training scenes made harder by their `harder` as the course chose on the validation scenes, and tests it once on
the 200 test scenes, made harder in the same way; after it, their `shares` shows where it fails on the 200 hard
validation scenes, where what to try next is chosen. The scenes are the course's own (scenes.py), so nothing is
downloaded."""
import dataclasses
import html
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import chart, live, netview, paths, projects, scenes, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_scenes_project",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_UNET = _lesson_checks("cv-unet")                 # the course's U-Net, and the mean IoU it is measured with
_FCN = _UNET._FCN                                 # the class maps of the scenes, how a U-Net trains, pictures of maps
_ONE = _lesson_checks("cv-one-stage-detector")    # the course's detector, how it trains, its mAP and its pictures

# The demo uses the learner's modules of the network they choose: for the U-Net, their class_map and kind_iou of cv-fcn
# and their DoubleConv, UNet and mean_iou of cv-unet; for the detector, their iou_matrix, match and boxes_from of
# cv-anchors and their targets, detection_loss and Detector of cv-one-stage-detector. Where one is missing or broken,
# the course's version stands in (template 1.3). Loading the two lessons above ran their own builds_on: this one comes
# after them.
BUILDS_ON = (("cnn", "classes", "cv-fcn", "class-map"), ("cnn", "kind_iou", "cv-fcn", "kind-iou"),
             ("cnn", "double_conv", "cv-unet", "double-conv"), ("cnn", "unet", "cv-unet", "unet"),
             ("cnn", "mean_iou", "cv-unet", "mean-iou"),
             ("cnn", "match", "cv-anchors", "match"), ("cnn", "offsets", "cv-anchors", "boxes-from"),
             ("cnn", "targets", "cv-one-stage-detector", "targets"),
             ("cnn", "detection_loss", "cv-one-stage-detector", "detection-loss"),
             ("cnn", "detector", "cv-one-stage-detector", "detector"))
projects.builds_on(*BUILDS_ON)

number, whole = _FCN.number, _FCN.whole
KINDS, CLASSES = _FCN.KINDS, _FCN.CLASSES   # circle, square and triangle; with background, 4 classes
SIZE = scenes.SIZE
TRAIN = range(1000, 3000)     # the training scenes, as in every lesson of this part
VAL = range(3000, 3200)       # the validation scenes: 200 others, to choose on
TEST = range(200)             # the test scenes, as in every lesson of this part: measured once, at the end
FOG, LIGHT, NOISE = 0.8, 0.8, 24.0   # a hard scene draws its fog, light and noise from 0 up to these
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # the levels of the slider: every change at this share of its largest
SETTINGS = ("easy", "both", "hard")   # what the network learns from: the scenes as they are, half made harder, all
MODELS = ("unet", "detector")
EPOCHS = 8                    # as in cv-unet and cv-one-stage-detector: both train for 8 epochs
SHOWN = (3001, 3003, 3004)    # the validation scenes the demo shows while it trains: shade, shade and fog, fog
EXAMPLE = 3003                # the validation scene on which it shows the layer it watches
PLAY = 3002                   # the validation scene of the slider
LIMITS = {"unet": 1800, "detector": 600}   # seconds the demo gives the training: the notes say thirty and ten minutes
SCORE = _ONE.SCORE            # a box of the detector counts when its score is 0.5 or more, as in its lesson
# What the groups of shapes are cut by: the pixels of a shape in sight, the light where it lies (1: as bright as in an easy scene),
# and the fog and the noise of its scene; each with the edges of its 4 bins.
GROUPS = {"size": (60, 250, 500, 1000, 1600), "light": (0.2, 0.4, 0.6, 0.8, 1.0), "fog": (0.0, 0.2, 0.4, 0.6, 0.8),
          "noise": (0.0, 6.0, 12.0, 18.0, 24.0)}


# --- harder scenes ----------------------------------------------------------------------------------------------------

def conditions(seed: int, level: float | None = None) -> dict:
    """How hard the scene of this seed is made: its fog, from 0 to FOG, its light, from 0 to LIGHT, and its noise, a
    number for every pixel and channel, normal with a standard deviation from 0 to NOISE gray values, all drawn from
    the seed. With `level`, every change is that share of its largest instead, with the same pattern of noise."""
    rng = np.random.default_rng(10_000 + seed)
    u = rng.uniform(0, 1, 3) if level is None else np.full(3, float(level))
    if level is not None:
        rng.uniform(0, 1, 3)   # the same draws, so the noise has the same pattern at every level
    sigma = NOISE * float(u[2])
    return {"fog": FOG * float(u[0]), "light": LIGHT * float(u[1]), "sigma": sigma,
            "noise": sigma * rng.standard_normal((SIZE, SIZE, 3))}


def course_harder(image, fog, light, noise):
    """A scene made harder: every number pulled to the mean of the whole image by the share `fog`, every column
    times 1 - light * (1 - j / (W - 1)), from 1 - light on the left to 1 on the right, and the noise added; then
    within 0 to 255, rounded, as whole numbers of uint8."""
    x = image.astype(float)
    m = x.mean()
    x = m + (x - m) * (1 - fog)
    w = x.shape[1]
    ramp = 1 - light * (1 - np.arange(w) / (w - 1))
    x = x * ramp.reshape(1, w, 1)
    x = x + noise
    return np.round(np.clip(x, 0, 255)).astype(np.uint8)


def hard_image(s: scenes.Scene, harder_fn=course_harder, level: float | None = None) -> np.ndarray:
    """The picture of scene s made harder by harder_fn, with the conditions of its seed."""
    c = conditions(s.seed, level)
    return np.asarray(harder_fn(s.image.copy(), c["fog"], c["light"], c["noise"].copy()))


def made(s: scenes.Scene, setting: str, harder_fn=course_harder) -> np.ndarray:
    """The picture a scene has in a setting: as it is ("easy"), made harder ("hard"), or made harder when its seed is
    even ("both": half of the training scenes)."""
    if setting == "easy" or setting == "both" and s.seed % 2:
        return s.image
    return hard_image(s, harder_fn)


_SCENES: dict = {}


def scene_list(which: str) -> tuple:
    """The training, validation or test scenes, made once."""
    if which not in _SCENES:
        _SCENES[which] = tuple(scenes.scene(seed) for seed in {"train": TRAIN, "val": VAL, "test": TEST}[which])
    return _SCENES[which]


def as_batch(images) -> torch.Tensor:
    """Pictures of (H, W, 3) as one tensor of bytes, shape (N, 3, H, W), as the networks of this part take them."""
    return torch.tensor(np.stack([np.asarray(i, dtype=np.uint8) for i in images])).permute(0, 3, 1, 2).contiguous()


def picture_set(which: str, setting: str, harder_fn=course_harder) -> torch.Tensor:
    """The pictures of the training, validation or test scenes in a setting, as one tensor of bytes."""
    return as_batch([made(s, setting, harder_fn) for s in scene_list(which)])


# --- where a network fails: every shape found or not -------------------------------------------------------------------

def unet_found(pred, ss) -> list[list[bool]]:
    """For every shape of every scene, whether the U-Net found it: more than half of its pixels in sight got its
    kind. pred: the class of every pixel, shape (N, H, W)."""
    out = []
    for p, s in zip(pred, ss):
        p = np.asarray(p)
        out.append([float((p[sh.mask] == sh.label + 1).mean()) > 0.5 for sh in s.shapes])
    return out


def detector_found(out, ss, boxes_fn=_ONE.boxes_from) -> list[list[bool]]:
    """For every shape of every scene, whether the detector found it: one of its boxes with a score of 0.5 or more,
    after NMS, has its kind and an IoU above 0.5 with it, and no box with a higher score found it first, as the mAP
    counts it."""
    found = []
    for i, s in enumerate(ss):
        shown = [d for d in _ONE.detect(out[i], boxes_fn=boxes_fn) if d[1] >= SCORE]
        got = set()
        for (kind, _, box), hit in zip(shown, _ONE.hits(shown, s)):
            if hit:   # the shape hits() gave it: the last one of its kind above the threshold
                got.add([q for q, sh in enumerate(s.shapes) if sh.label == kind and _ONE.iou(box, sh.box) > _ONE.AP_IOU][-1])
        found.append([q in got for q in range(len(s.shapes))])
    return found


def properties(ss) -> dict:
    """For every shape of the scenes ss, one after the other, what the groups are cut by: its pixels in sight, the light
    where its middle lies (the factor its column got), and the fog and the noise of its scene."""
    props = {"size": [], "light": [], "fog": [], "noise": []}
    for s in ss:
        c = conditions(s.seed)
        for sh in s.shapes:
            column = (sh.box[1] + sh.box[3]) / 2
            props["size"].append(int(sh.mask.sum()))
            props["light"].append(1 - c["light"] * (1 - column / (SIZE - 1)))
            props["fog"].append(c["fog"])
            props["noise"].append(c["sigma"])
    return props


def course_shares(found, values, edges):
    """For every bin of edges, the share of the shapes with a value in it that were found; None for an empty bin. A
    value v is in bin i when edges[i] <= v < edges[i + 1]."""
    found, values = np.asarray(found), np.asarray(values)
    out = []
    for i in range(len(edges) - 1):
        inside = (values >= edges[i]) & (values < edges[i + 1])
        out.append(float(found[inside].mean()) if inside.sum() > 0 else None)
    return out


def flat(found: list[list[bool]]) -> list[bool]:
    return [f for scene in found for f in scene]


def pack(found: list[bool]) -> str:
    return "".join("1" if f else "0" for f in found)


def unpack(text: str) -> list[bool]:
    return [c == "1" for c in text]


# --- the course's runs: how the settings were chosen, and what REFERENCE holds -----------------------------------------

def _course_targets(ss) -> tuple:
    return _ONE.all_targets(ss)


def class_maps(ss) -> torch.Tensor:
    return torch.tensor(np.stack([_FCN.class_map(s) for s in ss]), dtype=torch.int64)


def scored(model_kind: str, model, X, ss, C=None, boxes_fn=_ONE.boxes_from, miou_fn=_UNET.mean_iou) -> dict:
    """What a trained network does on the scenes ss with pictures X: for the U-Net, its mean IoU ("score") and the IoU
    of every class; for the detector, its mAP at an IoU of 0.5 ("score") and of 0.75, and the AP of every kind; for
    both, every shape found or not ("found")."""
    if model_kind == "unet":
        pred = _FCN.outputs(model, X).argmax(1)
        C = class_maps(ss) if C is None else C
        return {"score": float(miou_fn(pred, C, CLASSES)), "ious": _UNET.class_ious(pred, C), "found": unet_found(pred, ss),
                "pred": pred}
    out = _ONE.outputs(model, X)
    found_boxes = [_ONE.detect(out[i], boxes_fn=boxes_fn) for i in range(len(ss))]
    m, aps = _ONE.mean_ap(found_boxes, ss)
    m75, _ = _ONE.mean_ap(found_boxes, ss, 0.75)
    return {"score": float(m), "aps": [float(a) for a in aps], "map75": float(m75), "found": detector_found(out, ss, boxes_fn),
            "out": out}


def course_run(model_kind: str, setting: str, seed: int, heard=None) -> dict:
    """The course's network of this kind, trained on the 2,000 training scenes in this setting with `seed`, on one
    thread, as the demo trains: its loss after every epoch; on the validation scenes made harder its score and every
    shape found or not, and its score at every level of the slider; on the test scenes made harder and as they are,
    its score, and the shapes found on the hard ones. REFERENCE holds what it gave."""
    import time
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_ss, val_ss, test_ss = scene_list("train"), scene_list("val"), scene_list("test")
        X, Xv = picture_set("train", setting), picture_set("val", "hard")
        start = time.time()
        torch.manual_seed(seed)
        if model_kind == "unet":
            model = _UNET.make("unet")
            history = _FCN.train(model, (X, class_maps(train_ss)), (Xv, class_maps(val_ss)), seed, EPOCHS, heard)
        else:
            model = _ONE.Detector()
            history = _ONE.train(model, _ONE.detection_loss, (X, *_course_targets(train_ss)), (Xv, *_course_targets(val_ss)),
                                 seed, EPOCHS, heard)
        secs = time.time() - start
        val = scored(model_kind, model, Xv, val_ss)
        hard = scored(model_kind, model, picture_set("test", "hard"), test_ss)
        easy = scored(model_kind, model, picture_set("test", "easy"), test_ss)
        levels = [scored(model_kind, model, as_batch([hard_image(s, level=lv) for s in val_ss]), val_ss)["score"]
                  for lv in LEVELS]
    finally:
        torch.set_num_threads(threads)
    keep = ("ious",) if model_kind == "unet" else ("aps", "map75")
    return {"model": model_kind, "setting": setting, "seed": seed, "secs": round(secs),
            "curves": {"train": [round(float(h["train_loss"]), 4) for h in history],
                       "val": [round(float(h["val_loss"]), 4) for h in history]},
            "val": round(val["score"], 4), "val_found": sum(flat(val["found"])),
            "test": round(hard["score"], 4), "test_found": sum(flat(hard["found"])),
            "test_easy": round(easy["score"], 4), "levels": [round(v, 4) for v in levels],
            "found": pack(flat(val["found"])), **{k: [round(v, 4) for v in hard[k]] if isinstance(hard[k], list) else round(hard[k], 4) for k in keep}}


# What course_run gave for each network, setting and seed 0 to 4, one thread (made with the scratch script that
# runs course_run, two or three runs at a time): the score on the 200 hard validation scenes and the shapes found
# there, and on them as they are (level 0 of the slider); on the 200 test scenes made harder and as they are; the
# seconds of training; and, with seed 0, the score on the validation scenes at every level of the slider. For the
# setting the course chose (`chosen`), with seed 0, also the loss after every epoch, the IoU of every class or the AP
# of every kind and the mAP at 0.75 on the hard test scenes, and every shape of the hard validation scenes found or
# not.
REFERENCE = {
    "unet": {
        "val": {"easy": [0.3235, 0.1922, 0.3649, 0.233, 0.1794], "both": [0.8899, 0.5302, 0.8563, 0.8856, 0.8642], "hard": [0.8923, 0.8977, 0.8615, 0.9052, 0.8851]},
        "val_found": {"easy": [486, 409, 536, 404, 417], "both": [745, 411, 719, 743, 722], "hard": [746, 749, 720, 757, 744]},
        "test": {"easy": [0.3117, 0.1883, 0.3311, 0.2206, 0.1709], "both": [0.9057, 0.5493, 0.8846, 0.893, 0.8828], "hard": [0.9079, 0.9076, 0.8835, 0.9108, 0.903]},
        "test_found": {"easy": [517, 457, 519, 422, 440], "both": [759, 452, 752, 757, 747], "hard": [764, 762, 740, 767, 763]},
        "test_easy": {"easy": [0.9527, 0.9487, 0.9526, 0.6523, 0.9512], "both": [0.9257, 0.5733, 0.9164, 0.928, 0.915], "hard": [0.9221, 0.9205, 0.902, 0.9287, 0.9187]},
        "secs": {"easy": [605, 558, 548, 559, 569], "both": [537, 553, 558, 571, 612], "hard": [553, 575, 549, 601, 560]},
        "val_easy": {"easy": [0.9409, 0.9344, 0.9406, 0.6414, 0.9395], "both": [0.9123, 0.5571, 0.8901, 0.9159, 0.909], "hard": [0.9031, 0.9091, 0.8816, 0.9147, 0.8992]},   # level 0
        "levels": {"easy": [0.9409, 0.927, 0.3239, 0.1542, 0.1134], "both": [0.9123, 0.9137, 0.9059, 0.863, 0.4427], "hard": [0.9031, 0.9087, 0.9061, 0.8659, 0.5154]},   # seed 0
        # the setting the course chose, "hard", with seed 0: its loss after every epoch; on the hard test scenes,
        # its IoU of every class or AP of every kind; and every shape of the hard validation scenes found or not
        "curves": {"train": [0.8135, 0.1703, 0.13, 0.0843, 0.0546, 0.044, 0.0365, 0.0335],
                   "val": [1.0856, 0.8149, 0.1168, 1.5076, 0.0563, 0.1703, 0.0385, 0.0372]},
        "ious": [0.9977, 0.9312, 0.8446, 0.8581],
        "found": ("0101011111111110011111111111110111111111111111111111111111111111011110111111111111111111111111101110"
                  "1111111111111111111111011111111111111110111011111111111111111111111111111111111110111111111111111111"
                  "1111111111111111101111111111111111111111111101111111111011111110111111111111111111111111111111111111"
                  "1111111110111111111110111111111111010111110111101111111111111111111111111111111111111111111111111101"
                  "1111011111111011111111011101111111111111111111110111111111111111111111111111111111111101011011111111"
                  "1111111111111111111111111111111111110110111011111111111111111111111110101111111111110111100111111111"
                  "1111111111110101111111111111111111111110111111011111111111111111110110111111111110111011111111111111"
                  "1001111111111111111111111111111111111101111101111111111111111111101111110111011111111111111111111111"
                  "11"),
    },
    "detector": {
        "val": {"easy": [0.2839, 0.2576, 0.2231, 0.2817, 0.2805], "both": [0.7923, 0.801, 0.8096, 0.8026, 0.783], "hard": [0.8204, 0.7648, 0.743, 0.8146, 0.7755]},
        "val_found": {"easy": [313, 301, 241, 302, 266], "both": [604, 600, 612, 612, 578], "hard": [636, 596, 554, 635, 609]},
        "test": {"easy": [0.3148, 0.2729, 0.2643, 0.3067, 0.3156], "both": [0.8147, 0.7994, 0.8022, 0.8306, 0.7812], "hard": [0.8264, 0.7821, 0.7622, 0.8269, 0.7734]},
        "test_found": {"easy": [339, 322, 270, 322, 310], "both": [629, 605, 623, 632, 592], "hard": [646, 615, 579, 639, 608]},
        "test_easy": {"easy": [0.9159, 0.9119, 0.9137, 0.9182, 0.9059], "both": [0.8518, 0.8568, 0.8748, 0.8753, 0.847], "hard": [0.8684, 0.817, 0.786, 0.8599, 0.8333]},
        "secs": {"easy": [116, 118, 119, 118, 111], "both": [105, 115, 112, 107, 109], "hard": [100, 81, 92, 105, 102]},
        "val_easy": {"easy": [0.9214, 0.9119, 0.9042, 0.9117, 0.8975], "both": [0.8531, 0.8606, 0.8565, 0.8742, 0.836], "hard": [0.8507, 0.8024, 0.7687, 0.853, 0.8151]},   # level 0
        "levels": {"easy": [0.9214, 0.8501, 0.3717, 0.1047, 0.0054], "both": [0.8531, 0.8435, 0.8363, 0.7734, 0.3664], "hard": [0.8507, 0.8484, 0.8412, 0.7943, 0.404]},   # seed 0
        # the setting the course chose, "both", with seed 0: its loss after every epoch; on the hard test scenes,
        # its IoU of every class or AP of every kind; and every shape of the hard validation scenes found or not
        "curves": {"train": [0.3452, 0.1205, 0.1009, 0.0874, 0.0781, 0.0719, 0.0662, 0.0634],
                   "val": [0.1619, 0.1232, 0.1141, 0.0955, 0.0857, 0.0791, 0.0751, 0.0745]},
        "aps": [0.8842, 0.7225, 0.8374], "map75": 0.6988,
        "found": ("1101011111111110011111111111110110101111111110111110011101110011011111111111101111111011111101001110"
                  "1111111111111111001111011111111111111110111011010011100111110111111000111111111111011111111111111001"
                  "1110101101111101101111100111111111011011100001111110101111010111111001110111110111111111111111111011"
                  "0101110100101101110110101111111100110011011101100111011111111001011001101110101111111111111111111101"
                  "0110011011110111111100111101011011111001000111000111001111101111111111111111100111011101010001110111"
                  "0111111111111111011011101011111111100101011111110110110110111111011110101111111101000001000101011111"
                  "1111111111110000110111110100110111011110111101011111110101111111110100101111111110111110011110011111"
                  "1000011011110011111011111111001000111001111100110101111110110111100101110111001111111111011010111111"
                  "11"),
    },
}

# --- the exercises: shared ---------------------------------------------------------------------------------------------

REQUIRED = ["harder", "shares", "next-step", "report"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_F_EN = "With `{call}`, your function gives `{got}`, and it should be `{want}`"
_F_NL = "Met `{call}` geeft je functie `{got}`, en dat moet `{want}` zijn"


class _Named:
    """An argument of a case, written in the check's messages by its name, such as `image`; the function gets a copy
    of its value."""

    def __init__(self, shown: str, value):
        self.shown, self.value = shown, value

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return np.array(self.value, copy=True)


# --- exercise 1: harder ------------------------------------------------------------------------------------------------

_H_HEAD = "import numpy as np\n\n\ndef harder(image, fog, light, noise):\n"
_H_START = (_H_HEAD + "    # image: a scene, shape (H, W, 3), uint8; noise: a number for every pixel and channel, the same shape\n"
            "    # 1. fog: pull every number toward the mean m of the whole image: m + (x - m) * (1 - fog)\n"
            "    # 2. light: column j times 1 - light * (1 - j / (W - 1)): 1 - light on the left, 1 on the right\n"
            "    # 3. add the noise; keep every number within 0 and 255, round it, and give back uint8\n    ...\n")
_H_FLOAT = "    x = image.astype(float)\n"
_H_FOG = "    m = x.mean()\n    x = m + (x - m) * (1 - fog)\n"
_H_LIGHT = ("    w = x.shape[1]\n    ramp = 1 - light * (1 - np.arange(w) / (w - 1))\n"
            "    x = x * ramp.reshape(1, w, 1)\n")
_H_NOISE = "    x = x + noise\n"
_H_END = "    return np.round(np.clip(x, 0, 255)).astype(np.uint8)\n"


def _h(fog: str = _H_FOG, light: str = _H_LIGHT, noise: str = _H_NOISE, end: str = _H_END, first: str = _H_FLOAT) -> str:
    return _H_HEAD + first + fog + light + noise + end


_H = _h()
# The made-up scene of the worked example: 2 rows of 4 pixels, gray, with one red and one blue pixel, and its noise
_IMAGE = np.array([[[220, 220, 220], [220, 220, 220], [200, 60, 50], [220, 220, 220]],
                   [[220, 220, 220], [30, 140, 200], [220, 220, 220], [220, 220, 220]]], dtype=np.uint8)
_NOISE = np.random.default_rng(0).normal(0, 20, (2, 4, 3))
_TALL = _IMAGE.reshape(4, 2, 3).copy()   # the same pixels, 4 rows of 2


def _example_scene() -> tuple:
    """The validation scene of the slider as it is, with the fog, the light and the noise of its seed."""
    c = conditions(PLAY)
    return scenes.scene(PLAY).image, round(c["fog"], 2), round(c["light"], 2), c["noise"]


_PLAY_IMAGE, _PLAY_FOG, _PLAY_LIGHT, _PLAY_NOISE = _example_scene()
_H_VISIBLE = [(_Named("image", _IMAGE), 0.5, 0.6, _Named("noise", _NOISE))]
_H_HIDDEN = [
    (_Named("image", _IMAGE), 0.0, 0.0, _Named("np.zeros((2, 4, 3))", np.zeros((2, 4, 3)))),          # nothing changes
    (_Named("image", _IMAGE), 1.0, 0.0, _Named("np.zeros((2, 4, 3))", np.zeros((2, 4, 3)))),          # all fog: the mean
    (_Named("image", _IMAGE), 0.3, 1.0, _Named("np.zeros((2, 4, 3))", np.zeros((2, 4, 3)))),          # the left column black
    (_Named("image", _IMAGE), 0.2, 0.4, _Named("np.zeros((2, 4, 3)) + 60", np.zeros((2, 4, 3)) + 60)),   # above 255
    (_Named("image", _IMAGE), 0.2, 0.4, _Named("np.zeros((2, 4, 3)) - 150", np.zeros((2, 4, 3)) - 150)),   # below 0
    (_Named("image.reshape(4, 2, 3)", _TALL), 0.5, 0.6, _Named("np.zeros((4, 2, 3))", np.zeros((4, 2, 3)))),            # 4 rows of 2
    (_Named(f"scene({PLAY}).image", _PLAY_IMAGE), _PLAY_FOG, _PLAY_LIGHT, _Named(f"noise_{PLAY}", _PLAY_NOISE)),
]
_H_WRONG = {
    "no-clip": _h(end="    return np.round(x).astype(np.uint8)\n"),
    "no-round": _h(end="    return np.clip(x, 0, 255).astype(np.uint8)\n"),
    "float": _h(end="    return np.round(np.clip(x, 0, 255))\n"),
    "int": _h(end="    return np.round(np.clip(x, 0, 255)).astype(int)\n"),
    "fog-black": _h(fog="    x = x * (1 - fog)\n"),
    "fog-channels": _h(fog="    m = x.mean(axis=(0, 1))\n    x = m + (x - m) * (1 - fog)\n"),
    "dark-right": _h(light=_H_LIGHT.replace("(1 - np.arange(w) / (w - 1))", "np.arange(w) / (w - 1)")),
    "by-w": _h(light=_H_LIGHT.replace("/ (w - 1)", "/ w")),
    "rows": _h(light="    h = x.shape[0]\n    ramp = 1 - light * (1 - np.arange(h) / (h - 1))\n    x = x * ramp.reshape(h, 1, 1)\n"),
    "light-first": _H_HEAD + _H_FLOAT + _H_LIGHT + _H_FOG + _H_NOISE + _H_END,
    "no-noise": _h(noise=""),
}


def _image_text(v):
    """An array as the check's messages show it: as Python shows it when it is small, else its shape, dtype and mean."""
    a = np.asarray(v)
    if a.size <= 24:
        return " ".join(repr(a).split())
    if a.dtype.kind not in "biuf":
        return repr(a)
    mean = f"{a.mean():.2f}"
    return {"en": f"array of shape {tuple(a.shape)}, dtype {a.dtype}, mean {mean}",
            "nl": f"array met shape {tuple(a.shape)}, dtype {a.dtype}, gemiddelde {mean.replace('.', ',')}"}


def _same_image(got, want) -> bool:
    """The same numbers, of the same dtype, in the same shape."""
    try:
        g = np.asarray(got)
    except Exception:
        return False
    w = np.asarray(want)
    return g.shape == w.shape and g.dtype == w.dtype and bool(np.array_equal(g, w))


def _broadcast(e: BaseException) -> bool:
    return isinstance(e, ValueError) and "broadcast" in str(e)


_NOT_UINT8 = {"en": "With `{{call}}`, your numbers are right, but they are {kind}, not uint8. A picture of the course holds whole numbers of uint8: end with `.astype(np.uint8)`.",
              "nl": "Met `{{call}}` kloppen je getallen, maar het zijn {kind}, geen uint8. Een afbeelding van de course bevat hele getallen van uint8: eindig met `.astype(np.uint8)`."}
_H_MISTAKES = [
    Mistake("rows", error=_broadcast,
            en="With `{call}`, NumPy cannot multiply the image by your numbers of the light. The light changes from column to column: the columns are axis 1 of (H, W, 3), so there are `w = x.shape[1]` numbers, and `ramp.reshape(1, w, 1)` puts one in front of every column.",
            nl="Met `{call}` kan NumPy de afbeelding niet met je getallen van het licht vermenigvuldigen. Het licht verandert van kolom tot kolom: de kolommen zijn axis 1 van (H, W, 3), dus er zijn `w = x.shape[1]` getallen, en `ramp.reshape(1, w, 1)` zet er één voor elke kolom."),
    Mistake("no-clip", wrong=_H_WRONG["no-clip"],
            en=_F_EN + ": the numbers above 255 and below 0 do not fit in uint8, and `astype` does not stop them at 255 and 0. Keep every number within them first: `np.clip(x, 0, 255)`.",
            nl=_F_NL + ": de getallen boven 255 en onder 0 passen niet in uint8, en `astype` houdt ze niet bij 255 en 0 tegen. Houd elk getal er eerst binnen: `np.clip(x, 0, 255)`."),
    Mistake("no-round", wrong=_H_WRONG["no-round"],
            en=_F_EN + ": `astype` cuts off the decimals, so 99.6 becomes 99. Round first, with `np.round`, then 99.6 becomes 100.",
            nl=_F_NL + ": `astype` knipt de decimalen eraf, dus 99,6 wordt 99. Rond eerst af, met `np.round`, dan wordt 99,6 100."),
    Mistake("float", wrong=_H_WRONG["float"], en=_NOT_UINT8["en"].format(kind="floats"), nl=_NOT_UINT8["nl"].format(kind="floats")),
    Mistake("int", wrong=_H_WRONG["int"], en=_NOT_UINT8["en"].format(kind="whole numbers of another dtype"),
            nl=_NOT_UINT8["nl"].format(kind="hele getallen van een andere dtype")),
    Mistake("fog-black", wrong=_H_WRONG["fog-black"],
            en=_F_EN + ": your fog makes the image darker. Fog pulls every number toward the mean of the image, a gray: `m + (x - m) * (1 - fog)`; with fog 1, every number is the mean.",
            nl=_F_NL + ": je mist maakt de afbeelding donkerder. Mist trekt elk getal naar het gemiddelde van de afbeelding, een grijs: `m + (x - m) * (1 - fog)`; met fog 1 is elk getal het gemiddelde."),
    Mistake("fog-channels", wrong=_H_WRONG["fog-channels"],
            en=_F_EN + ": you pull every channel to its own mean, so the fog gets a color. The fog is gray: one number, the mean of the whole image, `x.mean()`.",
            nl=_F_NL + ": je trekt elk channel naar zijn eigen gemiddelde, dus de mist krijgt een kleur. De mist is grijs: één getal, het gemiddelde van de hele afbeelding, `x.mean()`."),
    Mistake("dark-right", wrong=_H_WRONG["dark-right"],
            en=_F_EN + ": your image is darkest on the right. The light comes from the right: column 0 gets 1 - light, the last column 1, with `1 - light * (1 - j / (W - 1))`.",
            nl=_F_NL + ": je afbeelding is rechts het donkerst. Het licht komt van rechts: kolom 0 krijgt 1 - light, de laatste kolom 1, met `1 - light * (1 - j / (W - 1))`."),
    Mistake("by-w", wrong=_H_WRONG["by-w"],
            en=_F_EN + ": the last column is a little too dark. It is column W - 1, and it should get 1: divide by `w - 1`, not by `w`.",
            nl=_F_NL + ": de laatste kolom is iets te donker. Het is kolom W - 1, en die moet 1 krijgen: deel door `w - 1`, niet door `w`."),
    Mistake("rows", wrong=_H_WRONG["rows"],
            en=_F_EN + ": your light changes from row to row, top to bottom. It changes from column to column: the columns are axis 1 of (H, W, 3), `w = x.shape[1]`, and `ramp.reshape(1, w, 1)`.",
            nl=_F_NL + ": je licht verandert van rij tot rij, van boven naar onder. Het verandert van kolom tot kolom: de kolommen zijn axis 1 van (H, W, 3), `w = x.shape[1]`, en `ramp.reshape(1, w, 1)`."),
    Mistake("light-first", wrong=_H_WRONG["light-first"],
            en=_F_EN + ": you change the light before the fog. First the fog, with the mean of the image as it is; then the light; then the noise.",
            nl=_F_NL + ": je verandert het licht vóór de mist. Eerst de mist, met het gemiddelde van de afbeelding zoals hij is; dan het licht; dan de ruis."),
    Mistake("no-noise", wrong=_H_WRONG["no-noise"],
            en=_F_EN + ": the noise is missing. Add it after the light: `x = x + noise`.",
            nl=_F_NL + ": de ruis ontbreekt. Tel hem op na het licht: `x = x + noise`."),
]


def harder_picture(seen) -> str:
    """What the learner's `harder` makes of the validation scene of the slider: the scene, then the scene with only
    fog, only light and only noise at their largest, and with all three of its own; with the course's version under
    any that differs. Nothing when their function does not run."""
    mine = seen.mine
    fn = mine.names.get("harder") if mine else None
    if not callable(fn):
        return ""
    image, own = scenes.scene(PLAY).image, conditions(PLAY)
    full = conditions(PLAY, 1.0)["noise"]
    zeros = np.zeros_like(full)
    lang = seen.lang
    shown = [((FOG, 0.0, zeros), {"en": f"fog {FOG}", "nl": f"mist {FOG}"}[lang]),
             ((0.0, LIGHT, zeros), {"en": f"light {LIGHT}", "nl": f"licht {LIGHT}"}[lang]),
             ((0.0, 0.0, full), {"en": f"noise σ {NOISE:g}", "nl": f"ruis σ {NOISE:g}"}[lang]),
             ((own["fog"], own["light"], own["noise"]), {"en": "all three, as drawn", "nl": "alle drie, zoals getrokken"}[lang])]
    words = {"en": ("as it is", "right"), "nl": ("zoals ze is", "goed")}[lang]
    cells = [_cell(image, [words[0]])]
    for (fog, light, noise), label in shown:
        try:
            got = with_time_limit(lambda: fn(image.copy(), fog, light, noise.copy()), 5)
            got = np.asarray(got)
        except Exception:
            return ""
        if got.shape != image.shape:
            return ""
        want = course_harder(image.copy(), fog, light, noise.copy())
        if label != words[0] and lang == "nl":
            label = label.replace(".", ",")
        cells.append(_cell(np.clip(got, 0, 255).astype(np.uint8), [label], None if _same_image(got, want) else (want, words[1])))
    title = {"en": f"Your `harder` on validation scene {PLAY}", "nl": f"Je `harder` op validatiescène {PLAY}"}[lang]
    body = '<div style="display:flex;flex-wrap:wrap;gap:10px 14px;align-items:flex-start">' + "".join(cells) + "</div>"
    return yours.figure(title, body, alt=title)


def _cell(image, label: list, right: tuple | None = None) -> str:
    """One scene drawn at 108 pixels, with the lines of its label above it, and the right one under it if given."""
    mono = "font-family:var(--ea-font-code);font-size:12px;color:var(--ea-ink-2)"
    parts = [f'<span style="{mono}">{html.escape(line)}</span>' for line in label] + [yours._png(np.asarray(image), 108) or ""]
    if right is not None:
        parts += [f'<span style="{mono}">{html.escape(right[1])}</span>', yours._png(np.asarray(right[0]), 108) or ""]
    return '<div style="display:flex;flex-direction:column;gap:3px">' + "".join(parts) + "</div>"


# --- exercise 2: shares --------------------------------------------------------------------------------------------------

_S_HEAD = "import numpy as np\n\n\ndef shares(found, values, edges):\n"
_S_START = (_S_HEAD + "    # found: for every shape whether the network found it; values: a number for every shape\n"
            "    # edges: k + 1 numbers from low to high; a value v is in bin i when edges[i] <= v < edges[i + 1]\n"
            "    # give back a list of k: for every bin the share of its shapes that was found, None for an empty bin\n"
            "    found, values = np.array(found), np.array(values)\n    ...\n")
_S_LOOP = "    out = []\n    for i in range(len(edges) - 1):\n"
_S_INSIDE = "        inside = (values >= edges[i]) & (values < edges[i + 1])\n"
_S_ADD = ("        if inside.sum() > 0:\n            out.append(float(found[inside].mean()))\n"
          "        else:\n            out.append(None)\n")
_S_END = "    return out\n"


def _s(loop: str = _S_LOOP, inside: str = _S_INSIDE, add: str = _S_ADD, end: str = _S_END) -> str:
    return _S_HEAD + "    found, values = np.array(found), np.array(values)\n" + loop + inside + add + end


_S = _s()
_FOUND = [True, False, True, True, False, True, False]     # the made-up shapes of the worked example
_SIZES = [80, 120, 300, 450, 90, 700, 1200]
_S_VISIBLE = [(_Named("found", np.array(_FOUND)), _Named("sizes", np.array(_SIZES)), [60, 250, 500, 1000, 1600])]
_S_HIDDEN = [
    ([True, False, True, False], [250, 250, 500, 60], [60, 250, 500, 1000]),     # values on the edges, as lists
    ([True, True, False], [0.1, 0.15, 0.7], [0.0, 0.2, 0.4, 0.6, 0.8]),          # empty bins
    ([True, False, True], [5, 300, 2000], [60, 250, 500, 1000, 1600]),            # values outside every bin
    ([True, True, True, True], [1.0, 2.0, 3.0, 4.0], [0, 2, 4, 6]),               # all found
]
_S_WRONG = {
    "of-all": _s(add=_S_ADD.replace("float(found[inside].mean())", "float(found[inside].sum() / len(found))")),
    "count": _s(add=_S_ADD.replace("float(found[inside].mean())", "int(found[inside].sum())")),
    "missed": _s(add=_S_ADD.replace("float(found[inside].mean())", "float(1 - found[inside].mean())")),
    "closed": _s(inside=_S_INSIDE.replace("values < edges[i + 1]", "values <= edges[i + 1]")),
    "zero": _s(add=_S_ADD.replace("out.append(None)", "out.append(0.0)")),
    "nan": _s(add="        out.append(float(found[inside].mean()))\n"),
}


def _same_shares(got, want) -> bool:
    """The same list of shares: None where the course has None, and every share within 1 in 10^9 (nan as nan)."""
    if isinstance(got, np.ndarray):
        got = got.tolist()
    if not isinstance(got, (list, tuple)) or len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if w is None or g is None:
            if g is not w:
                return False
            continue
        try:
            g, w = float(g), float(w)
        except (TypeError, ValueError):
            return False
        if math.isnan(w) or math.isnan(g):
            if not (math.isnan(w) and math.isnan(g)):
                return False
        elif abs(g - w) > 1e-9:
            return False
    return True


_S_MISTAKES = [
    Mistake("bins", error=lambda e: isinstance(e, IndexError),
            en="With `{call}`, your function stops: it reads past the last edge. k + 1 edges make k bins, so `i` goes from 0 to `len(edges) - 2`: `for i in range(len(edges) - 1)`.",
            nl="Met `{call}` stopt je functie: hij leest voorbij de laatste rand. k + 1 randen maken k bins, dus `i` gaat van 0 tot `len(edges) - 2`: `for i in range(len(edges) - 1)`."),
    Mistake("of-all", wrong=_S_WRONG["of-all"],
            en=_F_EN + ": you divide by all shapes. The share of a bin counts only its own shapes: the mean of `found` over the shapes in it, `found[inside].mean()`.",
            nl=_F_NL + ": je deelt door alle vormen. Het aandeel van een bin telt alleen zijn eigen vormen: het gemiddelde van `found` over de vormen erin, `found[inside].mean()`."),
    Mistake("count", wrong=_S_WRONG["count"],
            en=_F_EN + ": you give how many shapes of a bin were found. Give the share: how many found, divided by how many there are in the bin, `found[inside].mean()`.",
            nl=_F_NL + ": je geeft hoeveel vormen van een bin gevonden zijn. Geef het aandeel: hoeveel gevonden, gedeeld door hoeveel er in de bin zitten, `found[inside].mean()`."),
    Mistake("missed", wrong=_S_WRONG["missed"],
            en=_F_EN + ": you give the share that was missed. Give the share that was found: the mean of `found` in the bin.",
            nl=_F_NL + ": je geeft het aandeel dat gemist is. Geef het aandeel dat gevonden is: het gemiddelde van `found` in de bin."),
    Mistake("closed", wrong=_S_WRONG["closed"],
            en=_F_EN + ": a value on an edge is in two bins. A bin holds its lower edge and not its upper edge: `values < edges[i + 1]`.",
            nl=_F_NL + ": een waarde op een rand zit in twee bins. Een bin bevat zijn onderste rand en niet zijn bovenste: `values < edges[i + 1]`."),
    Mistake("zero", wrong=_S_WRONG["zero"],
            en=_F_EN + ": an empty bin gets 0. With no shapes in it, it has no share: give `None`, so that a bin without shapes is not read as a bin where none was found.",
            nl=_F_NL + ": een lege bin krijgt 0. Zonder vormen erin heeft hij geen aandeel: geef `None`, zodat een bin zonder vormen niet gelezen wordt als een bin waarin niets gevonden is."),
    Mistake("nan", wrong=_S_WRONG["nan"],
            en=_F_EN + ": an empty bin gets `nan`, the mean of nothing. Look first whether the bin holds a shape, with `inside.sum() > 0`, and give `None` when it does not.",
            nl=_F_NL + ": een lege bin krijgt `nan`, het gemiddelde van niets. Kijk eerst of de bin een vorm bevat, met `inside.sum() > 0`, en geef `None` als dat niet zo is."),
]


def _bin_labels(edges, lang: str) -> list[str]:
    def one(v):
        return f"{v:g}".replace(".", ",") if lang == "nl" else f"{v:g}"
    return [f"{one(a)}–{one(b)}" for a, b in zip(edges, edges[1:])]


def shares_chart(got: list, want: list | None, edges, title: str, xlabel: str, lang: str, alt: str, counts=None) -> str:
    """Shares per bin as bars, from 0 to 1, with `want` as a dashed line over each bar where given, and with `counts`
    the number of shapes of every bin above it; an empty bin gets no bar."""
    labels = _bin_labels(edges, lang)
    p = chart.Panel(x=(-0.6, len(labels) - 0.4), y=(0, 1.2 if counts else 1.08), title=title, xlabel=xlabel,
                    yticks=[0, 0.25, 0.5, 0.75, 1], xticks=list(range(len(labels))), xtick_labels=labels)
    for i, n in enumerate(counts or []):
        p.text(i, 1.12, f"n = {whole(n, lang)}", role="ink-2")
    for i, v in enumerate(got):
        if v is not None and math.isfinite(v):
            p.rect(i - 0.34, 0, i + 0.34, max(0.0, min(1.0, float(v))), role="data")
        if want is not None and want[i] is not None:
            p.line([i - 0.42, i + 0.42], [want[i]] * 2, role="ink", width=2, dashed=True)
    svg = chart.figure([p], alt=alt, width=276, height=210)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def shares_picture(seen) -> str:
    """What the learner's `shares` makes of the shapes of the 200 validation scenes, as the course's U-Net and detector,
    trained with seed 0 as the course chose, found them, cut by their pixels in sight; with the course's shares as a
    dashed line over every bar. Nothing when their function does not run."""
    mine = seen.mine
    fn = mine.names.get("shares") if mine else None
    if not callable(fn):
        return ""
    lang = seen.lang
    sizes = properties(scene_list("val"))["size"]
    names = {"unet": "U-Net", "detector": {"en": "detector", "nl": "detector"}[lang]}
    charts = []
    for model_kind in MODELS:
        found = unpack(REFERENCE[model_kind]["found"])
        try:
            got = with_time_limit(lambda: fn(list(found), list(sizes), list(GROUPS["size"])), 5)
            got = list(np.asarray(got, dtype=object)) if not isinstance(got, list) else got
            got = [None if g is None else float(g) for g in got]
        except Exception:
            return ""
        if len(got) != len(GROUPS["size"]) - 1:
            return ""
        want = course_shares(found, sizes, GROUPS["size"])
        words = {"en": (f"The course's {names[model_kind]}", "pixels in sight", "share found"),
                 "nl": (f"{'Het' if model_kind == 'unet' else 'De'} {names[model_kind]} van de course", "pixels in zicht", "aandeel gevonden")}[lang]
        charts.append((shares_chart(got, want, GROUPS["size"], words[0], words[1], lang, words[0]), 276))
    title = {"en": "Your `shares` on the shapes of the 200 validation scenes, made harder, cut by their pixels in sight; dashed: the right share",
             "nl": "Je `shares` op de vormen van de 200 validatiescènes, moeilijker gemaakt, verdeeld naar hun pixels in zicht; gestreept: het goede aandeel"}[lang]
    return yours.figure(title, _FCN.side_by_side(charts, title), alt=title)


EXERCISES = {
    "harder": FunctionBox(
        name="harder",
        picture=harder_picture,
        function="harder",
        start=_H_START,
        solution=_H,
        visible=_H_VISIBLE,
        hidden=_H_HIDDEN,
        compare=_same_image,
        show=_image_text,
        save_as=("cnn", "harder"),
        first=[_DOTS],
        mistakes=_H_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the fog, the light, the noise, and the numbers within 0 and 255 as uint8.",
                      "nl": "Vervang de `...` door je code: de mist, het licht, de ruis, en de getallen binnen 0 en 255 als uint8."},
            "wrong": {"en": _F_EN + ". Work with floats: first the fog, `m + (x - m) * (1 - fog)` with `m` the mean of the whole image; then every column `j` times `1 - light * (1 - j / (W - 1))`; then `+ noise`; and at the end `np.clip`, `np.round` and `.astype(np.uint8)`.",
                      "nl": _F_NL + ". Werk met floats: eerst de mist, `m + (x - m) * (1 - fog)` met `m` het gemiddelde van de hele afbeelding; dan elke kolom `j` keer `1 - light * (1 - j / (W - 1))`; dan `+ noise`; en aan het eind `np.clip`, `np.round` en `.astype(np.uint8)`."},
            "almost": {"en": "Right for the image of the worked example. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your function work with fog and light of 0 and of 1, with noise that pushes numbers above 255 and below 0, for an image with more rows than columns, and for a whole scene, " f"validation scene {PLAY} with the fog, the light and the noise its seed drew (`noise_{PLAY}`)?",
                       "nl": "Goed voor de afbeelding uit het uitgewerkte voorbeeld. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Werkt je functie met fog en light van 0 en van 1, met ruis die getallen boven 255 en onder 0 duwt, voor een afbeelding met meer rijen dan kolommen, en voor een hele scène, " f"validatiescène {PLAY} met de mist, het licht en de ruis die haar seed trok (`noise_{PLAY}`)?"},
            "correct": {"en": "Your `harder` is right: fog toward the gray of the mean, light that falls off to the left, noise, and whole numbers from 0 to 255. The shapes of a scene stay where they are; only their pixels change. It is now in your project, as `{saved}`.",
                        "nl": "Je `harder` klopt: mist naar het grijs van het gemiddelde, licht dat naar links afneemt, ruis, en hele getallen van 0 tot 255. De vormen van een scène blijven waar ze zijn; alleen hun pixels veranderen. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `harder` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `harder` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "shares": FunctionBox(
        name="shares",
        picture=shares_picture,
        function="shares",
        start=_S_START,
        solution=_S,
        visible=_S_VISIBLE,
        hidden=_S_HIDDEN,
        compare=_same_shares,
        save_as=("cnn", "shares"),
        first=[_DOTS],
        mistakes=_S_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: a loop over the bins, the shapes inside each one, and their share found.",
                      "nl": "Vervang de `...` door je code: een loop over de bins, de vormen in elke bin, en hun aandeel gevonden."},
            "wrong": {"en": _F_EN + ". For every bin `i`, from 0 to `len(edges) - 2`: the shapes with `edges[i] <= value < edges[i + 1]`, and the mean of `found` over them; `None` when there are none.",
                      "nl": _F_NL + ". Voor elke bin `i`, van 0 tot `len(edges) - 2`: de vormen met `edges[i] <= waarde < edges[i + 1]`, en het gemiddelde van `found` over die vormen; `None` als die er niet zijn."},
            "almost": {"en": "Right for the shapes of the worked example. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your function work for values on an edge, for empty bins, for values outside every bin, and for lists as well as arrays?",
                       "nl": "Goed voor de vormen uit het uitgewerkte voorbeeld. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Werkt je functie voor waarden op een rand, voor lege bins, voor waarden buiten elke bin, en voor lijsten net zo goed als arrays?"},
            "correct": {"en": "Your `shares` is right: for every bin the share of its shapes that was found, and `None` where there are none. It is now in your project, as `{saved}`; after the demo, it shows where your network fails.",
                        "nl": "Je `shares` klopt: voor elke bin het aandeel van zijn vormen dat gevonden is, en `None` waar die er niet zijn. Hij staat nu in je project, als `{saved}`; na de demo laat hij zien waar je netwerk faalt."},
            "kept": {"en": "Your `shares` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `shares` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("harder", _H_START, "empty", None),
    ("harder", _H, "correct", None),
    ("harder", _h(fog="    m = x.mean()\n    x = x * (1 - fog) + m * fog\n",
                  light="    w = x.shape[1]\n    ramp = (1 - light) + light * np.arange(w) / (w - 1)\n    x = x * ramp[None, :, None]\n"), "correct", None),
    ("harder", _h(first="    x = image / 1.0\n", end="    x = np.minimum(np.maximum(x, 0), 255)\n    return np.round(x).astype(np.uint8)\n"), "correct", None),
    ("harder", _h(light="    h, w = x.shape[0], x.shape[1]\n    ramp = 1 - light * (1 - np.arange(h) / (h - 1))\n    x = x * ramp.reshape(1, h, 1)\n"), "error", "rows"),
] + [("harder", code, "wrong", key) for key, code in _H_WRONG.items()] + [
    ("shares", _S_START, "empty", None),
    ("shares", _S, "correct", None),
    ("shares", _s(add="        n = int(inside.sum())\n        out.append(int(found[inside].sum()) / n if n > 0 else None)\n"), "correct", None),
    ("shares", _s(loop="    out = []\n    for lo, hi in zip(edges[:-1], edges[1:]):\n",
                  inside="        inside = (values >= lo) & (values < hi)\n"), "correct", None),
    ("shares", _s(loop="    out = []\n    for i in range(len(edges)):\n"), "error", "bins"),
    ("shares", _s(end="    return out[:2]\n"), "wrong", None),
] + [("shares", code, "wrong", key) for key, code in _S_WRONG.items()]


# --- the demo: the network the learner chooses, trained on scenes made harder by their `harder` --------------------------

_Stopped, _named = _FCN._Stopped, _FCN._named
# The learner's modules each network needs: (module, name, the key of the note when it stops)
MODULES = {"unet": (("classes", "class_map", "classes"), ("unet", "UNet", "unet"), ("mean_iou", "mean_iou", "mean_iou")),
           "detector": (("targets", "targets", "targets"), ("detection_loss", "detection_loss", "loss"),
                        ("detector", "Detector", "detector"), ("offsets", "boxes_from", "offsets"))}
# The modules of the network's own project lesson: the demo waits until they are saved; the others, of cv-fcn and
# cv-anchors, the course's versions stand in for when they are missing
OWN = {"unet": ("unet", "mean_iou"), "detector": ("targets", "detection_loss", "detector")}
# Every project module of cnn each network's demo runs: those above, and what they import
USES = {"unet": ("harder", "classes", "kind_iou", "double_conv", "unet", "mean_iou"),
        "detector": ("harder", "match", "offsets", "targets", "detection_loss", "detector")}
HARDER_LIMIT, TARGETS_LIMIT = 120, 240   # seconds for the 2,400 scenes made harder, and for the targets of 2,200 scenes
SCORE_LIMIT = 300             # seconds to measure the trained network on the validation and the test scenes
WORST = 3                     # the error analysis draws the 3 validation scenes with the most shapes missed


def chosen(model_kind: str) -> str:
    """The setting the course chose for this network: the one whose middle of five seeds is highest on the
    validation scenes (REFERENCE)."""
    return max(SETTINGS, key=lambda s: float(np.median(REFERENCE[model_kind]["val"][s])))


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _their_pictures(harder_fn, setting: str) -> dict:
    """The pictures the demo trains and measures on, made harder by the learner's `harder`: the training scenes in
    the setting, and the validation and test scenes made harder; and the test scenes as they are."""
    return {"train": picture_set("train", setting, harder_fn), "val": picture_set("val", "hard", harder_fn),
            "test": picture_set("test", "hard", harder_fn), "easy": picture_set("test", "easy")}


def _fits(harder_fn) -> bool:
    """Whether the learner's `harder` gives a picture of the scene of the slider: as large as the scene, with whole
    numbers from 0 to 255."""
    c = conditions(PLAY)
    got = np.asarray(harder_fn(scenes.scene(PLAY).image.copy(), c["fog"], c["light"], c["noise"].copy()))
    return got.shape == (SIZE, SIZE, 3) and got.dtype.kind in "iu" and bool(((got >= 0) & (got <= 255)).all())


def _image(X, i: int) -> np.ndarray:
    return X[i].permute(1, 2, 0).numpy()


def _worst(found: list[list[bool]]) -> list[int]:
    """The scenes with the most shapes missed, the most first (the lowest number first among equals)."""
    return sorted(range(len(found)), key=lambda i: (-(len(found[i]) - sum(found[i])), i))[:WORST]


def _demo(model_kind: str, show=None):
    """For the demo of the network model_kind ("unet" or "detector"): None when harder.py, or a module of that network's
    own project lesson (OWN), is not in the project yet; the key of the learner's module that stops or gives what does not fit ("harder",
    and for the U-Net "classes", "unet" or "mean_iou", for the detector "targets", "loss", "detector" or "offsets");
    "slow" when making the scenes harder, the targets, the training or the measuring takes too long; else a dict with what the demo
    shows: the setting the course chose, the loss after every epoch ("curves"), what the network makes of the SHOWN
    validation scenes after every epoch ("moments") with the layer it watches ("layers"), its score on the test scenes
    made harder ("score": mean IoU or mAP) and as they are ("easy"), and the shapes it found of the hard ones
    ("test_found" of "test_shapes"); every shape of the hard validation scenes found or not ("found"), the validation
    scenes with the most misses ("worst"), its parameters, the modules the course's versions stood in for, and
    the trained network layer by layer on the EXAMPLE ("net"). `show` hears the run while it trains, every report
    with "look": what the network makes at that moment (decision 1.17)."""
    if not all(_saved(module) for module in ("harder",) + OWN[model_kind]):
        return None
    needs = (("harder", "harder", "harder"),) + MODULES[model_kind]
    names = {}
    for module, name, key in needs:
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins() if module in USES[model_kind]]
    try:
        if not with_time_limit(lambda: _fits(names["harder"]), 10):
            return "harder"
        pics = with_time_limit(lambda: _their_pictures(names["harder"], chosen(model_kind)), HARDER_LIMIT)
    except Timeout:
        return "slow"
    except Exception:
        return "harder"
    run = (_demo_unet if model_kind == "unet" else _demo_detector)(names, pics, show)
    if isinstance(run, dict):
        run |= {"model": model_kind, "setting": chosen(model_kind), "stand_ins": stand_ins}
    return run


def _hearing(model, example_x, moments, layers, show, kept, looked):
    """How the demo hears the training: after every epoch it keeps what `kept` makes of the outputs on the
    validation scenes, and the layer it watches; every report goes to `show` with "look"."""
    def heard(p):
        out = p.get("pred", p.get("out"))
        if out is not None:
            moments.append(kept(out))
            layers.append(netview.watched(model, example_x))
            p = {k: v for k, v in p.items() if k not in ("pred", "out")}
        if show is not None:
            try:
                show(p | {"look": looked})
            except Timeout:   # the demo's time limit, also when it falls while the chart draws
                raise
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass
    return heard


def _demo_unet(names: dict, pics: dict, show):
    train_ss, val_ss, test_ss = scene_list("train"), scene_list("val"), scene_list("test")
    try:   # the learner's class maps: the targets of every scene
        C = _FCN._their_classes(names["class_map"], train_ss)
        Cv = _FCN._their_classes(names["class_map"], val_ss)
        Ct = _FCN._their_classes(names["class_map"], test_ss)
    except Exception:
        return "classes"
    if C is None or Cv is None or Ct is None:
        return "classes"
    try:   # the learner's U-Net on 2 scenes: 4 scores for every pixel, that training can follow back
        model = names["UNet"]()
        if not isinstance(model, torch.nn.Module):
            return "unet"
        out = model(pics["train"][:2].float() / 255)
        if not (torch.is_tensor(out) and tuple(out.shape) == (2, CLASSES, SIZE, SIZE) and out.requires_grad):
            return "unet"
    except Exception:
        return "unet"
    try:   # the learner's mean IoU of class maps compared with themselves: 1
        one = float(names["mean_iou"](Ct[:4], Ct[:4], CLASSES))
        if not (math.isfinite(one) and abs(one - 1) < 1e-6):
            return "mean_iou"
    except Exception:
        return "mean_iou"
    torch.manual_seed(0)   # the same start as the course's run, and batch normalization that saw no scene yet
    try:
        model = names["UNet"]()
    except Exception:
        return "unet"
    model.forward = _named(model.forward, "unet")   # an error in their U-Net is theirs; one in the course's code is not
    shown_at = [VAL.index(seed) for seed in SHOWN]
    shown = [_image(pics["val"], i) for i in shown_at]   # the SHOWN validation scenes, made harder
    example_x = pics["val"][VAL.index(EXAMPLE):VAL.index(EXAMPLE) + 1].float() / 255
    moments, layers = [], []

    def look() -> dict:
        """The class maps of the SHOWN validation scenes, and the layer the demo watches, at this moment, looked at in
        eval mode without a graph, so the run stays the same."""
        with netview.looking(model):
            maps = model(pics["val"][shown_at].float() / 255).argmax(1)
        return {"maps": [m.numpy() for m in maps], "layer": netview.watched(model, example_x), "images": shown}

    heard = _hearing(model, example_x, moments, layers, show, lambda pred: [pred[i].numpy() for i in shown_at], look)
    try:   # with_time_limit trains on one thread, as the course's run
        history = with_time_limit(lambda: _FCN.train(model, (pics["train"], C), (pics["val"], Cv), 0, EPOCHS, heard), LIMITS["unet"])
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if not all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history):
        return "unet"
    miou = _named(names["mean_iou"], "mean_iou")
    try:   # on one thread, as the course's run: on the validation scenes every shape, on the test scenes the score
        val, hard, easy = with_time_limit(lambda: (scored("unet", model, pics["val"], val_ss, C=Cv, miou_fn=miou),
                                                   scored("unet", model, pics["test"], test_ss, C=Ct, miou_fn=miou),
                                                   scored("unet", model, pics["easy"], test_ss, C=Ct, miou_fn=miou)), SCORE_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if not all(math.isfinite(r["score"]) for r in (val, hard, easy)):
        return "mean_iou"
    worst = _worst(val["found"])
    return {"curves": {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]},
            "moments": moments, "layers": layers, "score": hard["score"], "easy": easy["score"], "ious": hard["ious"],
            "test_found": sum(flat(hard["found"])), "test_shapes": len(flat(hard["found"])),
            "found": flat(val["found"]), "params": sum(p.numel() for p in model.parameters()),
            "worst": [(i, _image(pics["val"], i), val["pred"][i].numpy()) for i in worst],
            "shown": shown, "net": netview.trace(model, example_x)}


def _demo_detector(names: dict, pics: dict, show):
    train_ss, val_ss, test_ss = scene_list("train"), scene_list("val"), scene_list("test")
    try:
        (C, T), (Cv, Tv) = with_time_limit(lambda: (_ONE.all_targets(train_ss, names["targets"]),
                                                    _ONE.all_targets(val_ss, names["targets"])), TARGETS_LIMIT)
    except Timeout:
        return "slow"
    except Exception:
        return "targets"
    a = len(_ONE.anchors())
    if not all(c.shape == (len(ss), a) and t.shape == (len(ss), a, 4) for c, t, ss in ((C, T, train_ss), (Cv, Tv, val_ss))):
        return "targets"
    try:
        model = names["Detector"]()
        with torch.no_grad():
            if tuple(model(pics["train"][:2].float() / 255).shape) != (2, a, _ONE.PER):
                return "detector"
    except Exception:
        return "detector"
    try:
        model.train()
        loss = names["detection_loss"](model(pics["train"][:2].float() / 255), C[:2], T[:2])
        if not (torch.is_tensor(loss) and loss.numel() == 1 and torch.isfinite(loss) and loss.requires_grad):
            return "loss"
    except Exception:
        return "loss"
    torch.manual_seed(0)   # the same start as the course's run
    try:
        model = names["Detector"]()
    except Exception:
        return "detector"
    model.forward = _named(model.forward, "detector")   # an error in their detector is theirs; one in the course's code is not
    loss_fn = _named(names["detection_loss"], "loss")
    shown_at = [VAL.index(seed) for seed in SHOWN]
    shown = [_image(pics["val"], i) for i in shown_at]   # the SHOWN validation scenes, made harder
    example_x = pics["val"][VAL.index(EXAMPLE):VAL.index(EXAMPLE) + 1].float() / 255
    moments, layers, broken = [], [], []
    boxes_fn = _named(names["boxes_from"], "offsets")

    def kept(out) -> list:
        try:
            return [_ONE.shown_boxes(out[i], boxes_fn=boxes_fn) for i in shown_at]
        except Exception:
            broken.append(True)
            return [[] for _ in SHOWN]

    def look() -> dict:
        """The boxes on the SHOWN validation scenes, and the layer the demo watches, at this moment, looked at in eval
        mode without a graph, so the run stays the same."""
        with netview.looking(model):
            out = model(pics["val"][shown_at].float() / 255)
        return {"boxes": [_ONE.shown_boxes(o, boxes_fn=boxes_fn) for o in out], "layer": netview.watched(model, example_x),
                "images": shown}

    heard = _hearing(model, example_x, moments, layers, show, kept, look)
    try:
        history = with_time_limit(lambda: _ONE.train(model, loss_fn, (pics["train"], C, T), (pics["val"], Cv, Tv), 0, EPOCHS, heard),
                                  LIMITS["detector"])
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if broken:
        return "offsets"
    if not all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history):
        return "loss"
    try:   # on one thread, as the course's run: on the validation scenes every shape, on the test scenes the score
        val, hard, easy = with_time_limit(lambda: (scored("detector", model, pics["val"], val_ss, boxes_fn=boxes_fn),
                                                   scored("detector", model, pics["test"], test_ss, boxes_fn=boxes_fn),
                                                   scored("detector", model, pics["easy"], test_ss, boxes_fn=boxes_fn)), SCORE_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    worst = _worst(val["found"])
    return {"curves": {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]},
            "moments": moments, "layers": layers, "score": hard["score"], "easy": easy["score"], "aps": hard["aps"],
            "map75": hard["map75"], "test_found": sum(flat(hard["found"])), "test_shapes": len(flat(hard["found"])),
            "found": flat(val["found"]), "params": sum(p.numel() for p in model.parameters()),
            "worst": [(i, _image(pics["val"], i), _ONE.shown_boxes(val["out"][i], boxes_fn=boxes_fn)) for i in worst],
            "shown": shown, "net": netview.trace(model, example_x)}


_DEMO_RUNS: dict = {}


def _project_texts(model_kind: str) -> tuple:
    """The texts of the project files the demo of this network uses, as they are now (None for a missing file)."""
    texts = []
    for module in USES[model_kind]:
        path = projects.folder() / "cnn" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return (model_kind,) + tuple(texts)


def demo(model_kind: str, show=None):
    """The demo's result for this network, worked out once for the project files as they are; `show` hears the run
    when it trains."""
    key = _project_texts(model_kind)
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(model_kind, show)
    return _DEMO_RUNS[key]


def last_run():
    """The result of the last run of the demo, when it trained a network; else None."""
    for result in _DEMO_RUNS.values():
        if isinstance(result, dict):
            return result
    return None


# --- after the demo: where the network fails --------------------------------------------------------------------------

def course_found(model_kind: str) -> list[bool]:
    """Every shape of the hard validation scenes found or not by the course's network, trained as the course chose,
    seed 0."""
    return unpack(REFERENCE[model_kind]["found"])


def analysis(result: dict):
    """What the learner's `shares` makes of the shapes their network found and missed, for every group of GROUPS, beside
    the course's run with seed 0: None when shares.py is not in the project yet; "shares" when their `shares` stops or
    does not give a share or None for every bin; else a dict with their shares ("mine") and the course's ("course"),
    the number of shapes in every bin ("counts"), and what their run found and missed."""
    if not _saved("shares"):
        return None
    run = run_code("from cnn.shares import shares")
    if run.error is not None:
        return "shares"
    fn = run.names["shares"]
    props = properties(scene_list("val"))
    mine, course, counts = {}, {}, {}
    for key, edges in GROUPS.items():
        try:
            got = with_time_limit(lambda: fn(list(result["found"]), list(props[key]), list(edges)), 10)
            got = got.tolist() if isinstance(got, np.ndarray) else list(got)
            got = [None if g is None else float(g) for g in got]
        except Exception:
            return "shares"
        if len(got) != len(edges) - 1 or any(g is not None and not (0 <= g <= 1) for g in got):
            return "shares"
        mine[key] = got
        course[key] = course_shares(course_found(result["model"]), props[key], edges)
        values = np.asarray(props[key])
        counts[key] = [int(((values >= a) & (values < b)).sum()) for a, b in zip(edges, edges[1:])]
    return {"mine": mine, "course": course, "counts": counts, "found": sum(result["found"]), "shapes": len(result["found"])}


# --- pictures ------------------------------------------------------------------------------------------------------------

def _png64(image) -> str:
    """A picture of (H, W, 3) bytes as a PNG in base64, to draw inside an SVG."""
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_TOP = 22   # room for a title above a picture


def _rect(b, scale: float, role: str, width: float, dashed: bool) -> str:
    """A box (top, left, bottom, right) around its pixels, with an edge in the color of the page under it, so that it
    stands out on every shape, in light and in dark."""
    top, left, bottom, right = b
    x, y = left * scale, _TOP + top * scale
    w, h = (right - left + 1) * scale, (bottom - top + 1) * scale
    dash = "stroke-dasharray:4 2.5;" if dashed else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" style="fill:none;stroke:var(--ea-page);'
            f'stroke-width:{width + 2:.1f};opacity:0.85"/>'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" style="fill:none;stroke:var(--ea-{role});'
            f'stroke-width:{width:.1f};{dash}"/>')


def photo_svg(image, title: str, width: int = 120, boxes=()) -> str:
    """A picture of a scene, as large as `width`, with its title above it and boxes over it: each (box, role, line
    width, dashed), in pixels of the scene."""
    scale = width / image.shape[1]
    parts = [_FCN._title(title, width),
             f'<image href="data:image/png;base64,{_png64(image)}" x="0" y="{_TOP}" width="{width}" height="{width}" '
             'preserveAspectRatio="none" style="image-rendering:pixelated"/>']
    parts += [_rect(b, scale, role, w, dashed) for b, role, w, dashed in boxes]
    return _FCN._svg(parts, width, width + _TOP)


def _shown_panels(seen: dict, model_kind: str, words: dict) -> list[tuple[str, int]]:
    """What the network makes of the SHOWN validation scenes at one moment: for the U-Net, each scene made harder with
    its class map under it; for the detector, each scene made harder with its true boxes and the boxes of 0.5 or more
    the detector gives it."""
    panels = []
    for k, seed in enumerate(SHOWN):
        s, image = scenes.scene(seed), seen["images"][k]
        title = words["scene"].format(seed=seed)
        if model_kind == "unet":
            cell = ('<div style="display:flex;flex-direction:column;gap:0">' + photo_svg(image, title, 120)
                    + _FCN.map_svg(seen["maps"][k], words["map"], 120) + "</div>")
        else:
            found = seen["boxes"][k]
            boxes = [(sh.box, "data", 1.3, False) for sh in s.shapes] + [(b, "pred", 1.6, True) for _, _, b in found]
            cell = photo_svg(image, title, 120, boxes)
        panels.append((cell, 120))
    return panels


def now_pictures(seen: dict, model_kind: str, words: dict, lang: str, uid: str) -> list[str]:
    """What the learner's network makes at one moment of its run, as `look` or a rewind gives it, of the SHOWN
    validation scenes ("images", made harder by their `harder`), and under it the layer the demo watches, under
    words["layer"] with its name. `uid` keeps the masks of the layer apart from those of another picture."""
    shown = _FCN.side_by_side(_shown_panels(seen, model_kind, words), words["alt"])
    if model_kind == "unet":
        shown += _FCN.key_html(words["kinds"])
    else:
        shown += _line_key([(words["shapes"], "data", False), (words["boxes"], "pred", True)], 380)
    return netview.moment(shown, seen.get("layer"), words["layer"], lang, uid, seed=EXAMPLE)


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the training scenes solid and on the validation scenes
    dashed; with `at`, a line at that epoch."""
    return _ONE.curves_picture(curves, labels, lang, at)


def epoch_frame(result: dict, e: int, labels: dict, words: dict, lang: str) -> str:
    """The rewind at epoch e (from 1): the loss after every epoch with a line at e, beside what the network made of the
    SHOWN validation scenes after it, and under them the layer the demo watches, as it was then."""
    key = "maps" if result["model"] == "unet" else "boxes"
    seen = {key: result["moments"][e - 1], "layer": result["layers"][e - 1], "images": result["shown"]}
    return live.rewound(curves_picture(result["curves"], labels, lang, at=e),
                        now_pictures(seen, result["model"], words, lang, f"nv-scenes-project-epoch-{e}"))


def layer_frame(net, k, lang: str) -> str:
    """The slider over the layers of the learner's trained network: the network, with layer k (from 1) chosen and its
    feature maps under it."""
    return netview.picture(net, lang, chosen=int(k) - 1, uid=f"nv-scenes-project-{int(k)}")


def play_frame(level: float, labels: dict, lang: str) -> tuple[str, dict]:
    """The slider over the levels: the validation scene of the slider at `level`, one of LEVELS, beside the score of
    the course's U-Net and detector on the 200 validation scenes at every level, each trained on easy scenes and as
    the course chose (seed 0), with a line at this level; and the numbers the sentence under it says."""
    k = min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] - float(level)))
    level = LEVELS[k]
    c = conditions(PLAY, level)
    image = hard_image(scenes.scene(PLAY), level=level)
    p = chart.Panel((-0.04, 1.04), (0, 1.0), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=list(LEVELS), yticks=[0, 0.25, 0.5, 0.75, 1])
    stop = {"level": level, "fog": c["fog"], "light": c["light"], "noise": c["sigma"]}
    for model_kind, dashed in (("unet", False), ("detector", True)):
        for setting, role in (("easy", "ink-3"), (chosen(model_kind), "ink")):
            ys = REFERENCE[model_kind]["levels"][setting]
            p.line(list(LEVELS), ys, role, 2.0, dashed=dashed)
            p.points([level], [ys[k]], role, r=3.5)
            stop[f"{model_kind}_{'easy' if setting == 'easy' else 'chosen'}"] = ys[k]
    p.line([level, level], [0, 1.0], "ink-2", 1.0)
    plot = chart.figure([p], alt=labels["alt"], width=276, height=236)   # 11 px text in a play box on a phone
    if lang == "nl":
        plot = chart.with_decimal_comma(plot)
    key = _line_key([(labels[f"{m}_{s}"], "ink-3" if s == "easy" else "ink", m == "detector")
                     for m in MODELS for s in ("easy", "chosen")])
    scene = photo_svg(image, labels["scene"].format(seed=PLAY), 200)
    return _FCN.side_by_side([(scene, 200), (plot + key, 276)], labels["alt"]), stop


def _line_key(entries, width: int = 276) -> str:
    """A key of lines: each (name, role, dashed), as a short line in its style and its name, wrapped at `width`. Its
    size is in its style too: a play box, the live chart and the rewind size every svg in them to their width."""
    items = []
    for name, role, dashed in entries:
        dash = ' stroke-dasharray="5 3"' if dashed else ""
        items.append(f'<span style="white-space:nowrap;display:inline-flex;align-items:center;gap:0.3rem"><svg width="26" height="10" '
                     f'viewBox="0 0 26 10" aria-hidden="true" style="flex:none;display:inline-block;width:26px;height:10px">'
                     f'<line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-{role});stroke-width:2"{dash}/></svg>'
                     f'{html.escape(name)}</span>')
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;flex-wrap:wrap;'
            f'gap:0.2rem 1rem;margin:0 0 0.4rem;max-width:{width}px">' + "".join(items) + "</div>")


def groups_figure(found: dict, labels: dict, lang: str) -> str:
    """The share of the shapes of the hard validation scenes the learner's network found, for every group of GROUPS,
    as their `shares` gave it, with the share of the course's run (seed 0) as a dashed line over every bar, and the
    number of shapes of every group above it."""
    charts = []
    for key in GROUPS:
        charts.append((shares_chart(found["mine"][key], found["course"][key], GROUPS[key], labels[key], labels[f"{key}_x"],
                                    lang, labels["alt"].format(what=labels[key]), found["counts"][key]), 276))
    return _FCN.side_by_side(charts, labels["alt"].format(what=", ".join(labels[k] for k in GROUPS))) + _line_key(
        [(labels["mine"], "data", False), (labels["course"], "ink", True)])


def worst_picture(result: dict, words: dict, lang: str) -> str:
    """The WORST validation scenes of the learner's run, made harder: every shape it missed in a box in the color of
    mistakes, every shape it found in a thin box; for the U-Net, its class map under the scene, for the detector, its
    boxes of 0.5 or more, dashed; with a key of the boxes."""
    found = result["found"]
    starts = np.cumsum([0] + [len(s.shapes) for s in scene_list("val")])
    panels = []
    for i, image, made_of in result["worst"]:
        s = scene_list("val")[i]
        mine = found[starts[i]:starts[i + 1]]
        boxes = [(sh.box, "data" if f else "loss", 1.3 if f else 2.4, False) for sh, f in zip(s.shapes, mine)]
        missed = len(mine) - sum(mine)
        title = words["scene"][0 if missed == 1 else 1].format(seed=s.seed, n=missed)
        if result["model"] == "unet":
            cell = ('<div style="display:flex;flex-direction:column;gap:0">' + photo_svg(image, title, 150, boxes)
                    + _FCN.map_svg(made_of, words["map"], 150) + "</div>")
        else:
            cell = photo_svg(image, title, 150, boxes + [(b, "pred", 1.6, True) for _, _, b in made_of])
        panels.append((cell, 150))
    out = _FCN.side_by_side(panels, words["alt"])
    key = [(words["missed"], "loss", False), (words["found"], "data", False)]
    if result["model"] == "detector":
        key.append((words["boxes"], "pred", True))
    return out + _line_key(key, 420) + (_FCN.key_html(words["kinds"]) if result["model"] == "unet" else "")


def changes_picture(words: dict) -> str:
    """The validation scene of the slider as it is, and made harder by the course's `harder`: with only fog, only light
    and only noise at their largest, and with all three as its seed drew them."""
    s = scenes.scene(PLAY)
    own, full = conditions(PLAY), conditions(PLAY, 1.0)
    zeros = np.zeros_like(full["noise"])
    made_of = [(s.image, words["as_is"]),
               (course_harder(s.image, FOG, 0.0, zeros), words["fog"]),
               (course_harder(s.image, 0.0, LIGHT, zeros), words["light"]),
               (course_harder(s.image, 0.0, 0.0, full["noise"]), words["noise"]),
               (course_harder(s.image, own["fog"], own["light"], own["noise"]), words["all"])]
    return _FCN.side_by_side([(photo_svg(image, title, 120), 120) for image, title in made_of], words["alt"])
