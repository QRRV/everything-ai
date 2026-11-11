# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-focal-loss: focal_loss, the focal loss of the logits of every anchor that is not ignored,
summed and divided by the number of anchors that learn a shape, and background_bias, the bias of the background logit
with which every anchor starts at background with a probability of 1 - pi. Both are saved in the learner's project
(work/projects/cnn/focal.py and background.py) when right. The demo trains the course's FPN detector of cv-fpn with
them, and with the learner's pyramid, head_rows and boxes_from where those are there and right, on the course's scenes
with small and large shapes, and counts what it finds. And the pictures: the focal loss for several values of gamma,
and for the slider, the course's detector trained with each of them. Everything is computed from the course's own
scenes (scenes.py): no download."""
import functools
import importlib.util
import math
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F

from everything_ai import chart, paths, projects, scenes
from everything_ai.check import Mistake, ScenarioBox, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_focal",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The FPN detector, its anchors, scenes, training, detecting and pictures; and through it the one-stage detector's
_FPN = _lesson_checks("cv-fpn")
_ONE = _FPN._ONE

# The demo builds the FPN detector with the learner's pyramid and head_rows of cv-fpn, and finds the boxes with their
# boxes_from of cv-anchors; where one is missing or broken, the course's version stands in, and the lesson says so.
BUILDS_ON = (("cnn", "offsets", "cv-anchors", "boxes-from"), ("cnn", "pyramid", "cv-fpn", "pyramid"),
             ("cnn", "head_rows", "cv-fpn", "head-rows"))
projects.builds_on(*BUILDS_ON)

SEED = _FPN.SEED              # the scene of cv-fpn's worked example, scene(26, spread=True), a test scene
KINDS = _FPN.KINDS            # circle, square, triangle: 3 kinds besides background
GAMMA = 2.0                   # the focal loss of RetinaNet
PI = 0.01                     # every anchor starts at background with a probability of 1 - PI
EASY = 0.9                    # an anchor is easy when the detector gives what it learns a probability above this
GAMMAS = (0.0, 1.0, 2.0, 5.0)  # the values of gamma the slider shows, each a detector the course trained
DRAWN = (0.0, 0.5, 1.0, 2.0, 5.0)   # the curves of the picture, as in the paper's first figure
N_TRAIN, N_TEST, TRAIN, TEST = _FPN.N_TRAIN, _FPN.N_TEST, _FPN.TRAIN, _FPN.TEST
EPOCHS, BATCH = _FPN.EPOCHS, _FPN.BATCH
SHOWN, SHOWN_SHAPES = _FPN.SHOWN, _FPN.SHOWN_SHAPES   # the test scenes of the rewind
TRAIN_LIMIT = 600             # seconds the demo gives the training: the note says ten minutes
anchors, scene_set, course_targets, outputs = _FPN.anchors, _FPN.scene_set, _FPN.course_targets, _FPN.outputs
detect, shown, example, summary = _FPN.detect, _FPN.shown, _FPN.example, _FPN.summary
iou, hits, number, whole = _ONE.iou, _ONE.hits, _ONE.number, _ONE.whole

# The losses the course compares, each trained with the seeds 0 to 4: gamma, whether every anchor starts at
# background, and what the summed loss is divided by: the anchors that learn a shape, or every anchor that is not
# ignored (a mean). "mean" and "mean-start" are the loss of the one-stage detector, the mean of the cross-entropy.
LOSSES = {
    "focal": (2.0, True, "shapes"),
    "gamma0": (0.0, True, "shapes"),
    "gamma1": (1.0, True, "shapes"),
    "gamma5": (5.0, True, "shapes"),
    "no-start": (2.0, False, "shapes"),
    "ce-no-start": (0.0, False, "shapes"),
    "focal-mean": (2.0, True, "all"),
    "mean": (0.0, False, "all"),
    "mean-start": (0.0, True, "all"),
}
BY_GAMMA = {0.0: "gamma0", 1.0: "gamma1", 2.0: "focal", 5.0: "gamma5"}

# The numbers of the worked example, measured without training. `measure()` measures them again; the test of the
# lesson checks that both are the same.
REFERENCE = {'ce': [0.0074, 0.1392, 2.3408, 0.02, 1.0064], 'p': [0.9926, 0.87, 0.0963, 0.9802, 0.3655],
             'fl': [4.0365e-07, 0.0023508, 1.9118, 7.8562e-06, 0.40513], 'ratio': [18354.0, 59.216, 1.2244, 2547.3, 2.4841],
             'loss': [1.7569, 1.1597],
             'scene': {'background': 2693, 'ignored': 10, 'shapes': 17, 'ce': 1.3863, 'fl': 0.7798, 'ce_share': 0.9937,
                       'fl_share': 0.9937},
             'bias': 5.6937}

# What the course measured, on one thread, with the course's code (train_course): the FPN detector of cv-fpn after
# torch.manual_seed(seed), every anchor started at background or not, trained as cv-fpn trained it but with each loss
# of LOSSES, for the seeds 0 to 4. "map", "aps", "map75", "found", "false" and "val" as in cv-fpn's TRAINED. For seed 0
# of the losses with gamma and the start: "scene", its boxes of 0.5 or more on the scene of the worked example (kind,
# score, box, anchor); and "where", on the 200 test scenes, how many background anchors are easy (a probability of
# background above EASY), and which share of the summed cross-entropy and of the summed focal loss (gamma 2) comes
# from the easy background anchors, the other background anchors and the anchors that learn a shape. "curves" is the
# loss over every epoch of "focal" with seed 0. "hard" is cv-fpn's FPN detector with hard negative mining.
# tests/test_cv_focal_loss.py measures "focal" with seed 0 again; measure_trained() measures all of it.
TRAINED = {'focal': {'map': [0.949, 0.9442, 0.9496, 0.9447, 0.9507],
                     'aps': [[0.968, 0.9417, 0.9373],
                             [0.9465, 0.937, 0.9493],
                             [0.9774, 0.9404, 0.931],
                             [0.9567, 0.9288, 0.9486],
                             [0.961, 0.9363, 0.9547]],
                     'map75': [0.7775, 0.7653, 0.7687, 0.7924, 0.788],
                     'found': [[244, 261, 218], [247, 262, 219], [243, 262, 224], [249, 262, 223], [243, 266, 223]],
                     'false': [95, 98, 109, 96, 97],
                     'val': [0.3838, 0.384, 0.3699, 0.3541, 0.3702],
                     'scene': [(0, 1.0, (30.5, 42.0, 52.4, 63.6), 2221),
                               (0, 0.999, (71.8, 11.9, 90.6, 29.7), 2373),
                               (2, 0.998, (73.8, 65.0, 112.5, 103.2), 2651),
                               (0, 0.985, (95.4, 30.6, 105.0, 39.9), 1617),
                               (1, 0.884, (100.0, 107.3, 108.6, 116.2), 1719)],
                     'where': {'anchors': [537874, 1568, 2043],
                               'ce': [0.5601, 0.1456, 0.2943],
                               'fl': [0.0021, 0.1537, 0.8442],
                               'shapes_p': [0.7748, 0.8341]}},
           'gamma0': {'map': [0.9338, 0.9336, 0.9434, 0.9389, 0.942],
                      'aps': [[0.9582, 0.937, 0.9062],
                              [0.9547, 0.9211, 0.925],
                              [0.9731, 0.9275, 0.9298],
                              [0.9433, 0.9377, 0.9357],
                              [0.9544, 0.9413, 0.9301]],
                      'map75': [0.7504, 0.7417, 0.742, 0.7609, 0.7831],
                      'found': [[250, 260, 221], [249, 262, 218], [246, 268, 226], [250, 263, 224], [250, 263, 225]],
                      'false': [145, 127, 132, 133, 135],
                      'val': [0.8508, 0.8662, 0.8219, 0.8529, 0.8294],
                      'scene': [(0, 1.0, (31.7, 42.8, 54.0, 64.8), 2221),
                                (0, 1.0, (71.7, 12.1, 90.8, 30.4), 2373),
                                (2, 0.999, (73.3, 63.1, 115.8, 103.6), 2651),
                                (1, 0.997, (100.3, 106.2, 109.1, 115.5), 1721),
                                (0, 0.925, (95.9, 30.9, 105.0, 39.7), 1617),
                                (1, 0.624, (95.3, 29.8, 105.5, 39.9), 1553)],
                      'where': {'anchors': [539174, 268, 2043],
                                'ce': [0.0144, 0.1857, 0.7999],
                                'fl': [0.0, 0.1607, 0.8393],
                                'shapes_p': [0.8424, 0.8571]}},
           'gamma1': {'map': [0.9421, 0.9471, 0.9435, 0.9411, 0.9582],
                      'aps': [[0.9599, 0.9363, 0.93],
                              [0.9618, 0.9318, 0.9476],
                              [0.9602, 0.9332, 0.9371],
                              [0.9465, 0.9411, 0.9356],
                              [0.9685, 0.9416, 0.9645]],
                      'map75': [0.7683, 0.7797, 0.7788, 0.8063, 0.7824],
                      'found': [[244, 261, 223], [250, 263, 223], [243, 261, 222], [244, 263, 224], [250, 263, 228]],
                      'false': [115, 101, 127, 105, 100],
                      'val': [0.5781, 0.5449, 0.5218, 0.5103, 0.5074],
                      'scene': [(0, 1.0, (31.3, 42.0, 53.4, 63.9), 2221),
                                (0, 1.0, (71.9, 11.4, 90.7, 29.8), 2373),
                                (2, 0.998, (75.3, 64.6, 114.4, 103.3), 2651),
                                (0, 0.989, (96.2, 30.9, 104.8, 39.7), 1617),
                                (1, 0.949, (100.3, 106.5, 109.4, 115.9), 1721)],
                      'where': {'anchors': [538780, 662, 2043],
                                'ce': [0.1869, 0.2, 0.6131],
                                'fl': [0.0003, 0.1686, 0.8311],
                                'shapes_p': [0.8069, 0.8434]}},
           'gamma5': {'map': [0.9355, 0.9311, 0.9369, 0.9345, 0.9413],
                      'aps': [[0.9496, 0.9349, 0.9219],
                              [0.9457, 0.9283, 0.9193],
                              [0.958, 0.9391, 0.9138],
                              [0.9478, 0.9384, 0.9174],
                              [0.9565, 0.9407, 0.9266]],
                      'map75': [0.7923, 0.7864, 0.7999, 0.784, 0.7848],
                      'found': [[238, 253, 219], [241, 253, 213], [235, 252, 221], [236, 253, 217], [243, 253, 215]],
                      'false': [79, 81, 71, 75, 78],
                      'val': [0.1943, 0.2033, 0.1913, 0.1869, 0.1865],
                      'scene': [(0, 0.983, (32.1, 41.9, 53.6, 63.3), 2221),
                                (0, 0.951, (72.3, 11.8, 91.4, 30.3), 2372),
                                (2, 0.943, (73.3, 62.0, 114.4, 103.1), 2651),
                                (0, 0.844, (95.4, 30.5, 104.8, 39.7), 1617),
                                (1, 0.629, (100.3, 107.0, 108.7, 115.5), 1721)],
                      'where': {'anchors': [520377, 19065, 2043],
                                'ce': [0.7243, 0.2121, 0.0636],
                                'fl': [0.032, 0.3523, 0.6157],
                                'shapes_p': [0.6607, 0.7851]}},
           'no-start': {'map': [0.9416, 0.9424, 0.947, 0.9501, 0.9468],
                        'aps': [[0.9502, 0.9466, 0.9281],
                                [0.9534, 0.943, 0.9309],
                                [0.9593, 0.9481, 0.9337],
                                [0.9707, 0.9432, 0.9365],
                                [0.9608, 0.9525, 0.9272]],
                        'map75': [0.7829, 0.7717, 0.7686, 0.7656, 0.7746],
                        'found': [[245, 261, 223],
                                  [239, 261, 223],
                                  [243, 261, 224],
                                  [247, 258, 223],
                                  [250, 264, 221]],
                        'false': [108, 100, 105, 103, 113],
                        'val': [0.4007, 0.3775, 0.3976, 0.3897, 0.3932]},
           'ce-no-start': {'map': [0.9268, 0.9323, 0.9383, 0.9309, 0.9363],
                           'aps': [[0.9503, 0.9097, 0.9203],
                                   [0.9513, 0.9246, 0.9212],
                                   [0.9549, 0.9407, 0.9193],
                                   [0.948, 0.9266, 0.9181],
                                   [0.953, 0.9434, 0.9126]],
                           'map75': [0.7081, 0.7333, 0.7059, 0.7115, 0.7199],
                           'found': [[244, 259, 223],
                                     [247, 267, 223],
                                     [250, 257, 224],
                                     [246, 264, 220],
                                     [246, 261, 224]],
                           'false': [137, 124, 136, 126, 126],
                           'val': [0.8596, 0.8193, 0.8659, 0.8506, 0.8477]},
           'focal-mean': {'map': [0.6425, 0.6683, 0.6426, 0.6458, 0.644],
                          'aps': [[0.738, 0.5161, 0.6733],
                                  [0.7562, 0.517, 0.7317],
                                  [0.7457, 0.4533, 0.7288],
                                  [0.7572, 0.4841, 0.696],
                                  [0.7575, 0.521, 0.6535]],
                          'map75': [0.518, 0.5645, 0.5373, 0.5377, 0.5341],
                          'found': [[94, 78, 61], [111, 97, 78], [88, 81, 65], [103, 76, 66], [83, 78, 77]],
                          'false': [57, 66, 45, 58, 68],
                          'val': [0.0459, 0.045, 0.0447, 0.045, 0.0443]},
           'mean': {'map': [0.3334, 0.3936, 0.3633, 0.3729, 0.3687],
                    'aps': [[0.3821, 0.1877, 0.4306],
                            [0.4334, 0.2726, 0.475],
                            [0.4336, 0.2084, 0.4477],
                            [0.502, 0.2539, 0.363],
                            [0.403, 0.2778, 0.4253]],
                    'map75': [0.2768, 0.3224, 0.2863, 0.296, 0.2922],
                    'found': [[6, 2, 1], [11, 6, 1], [15, 4, 2], [25, 24, 10], [4, 3, 1]],
                    'false': [2, 2, 2, 34, 2],
                    'val': [0.05, 0.0508, 0.0507, 0.0501, 0.0496]},
           'mean-start': {'map': [0.6793, 0.6714, 0.6437, 0.6803, 0.6364],
                          'aps': [[0.7818, 0.4979, 0.7582],
                                  [0.7912, 0.4783, 0.7448],
                                  [0.7647, 0.42, 0.7464],
                                  [0.7864, 0.4876, 0.767],
                                  [0.7581, 0.4646, 0.6866]],
                          'map75': [0.5635, 0.5706, 0.5567, 0.5773, 0.5418],
                          'found': [[133, 137, 97],
                                    [139, 146, 130],
                                    [128, 122, 111],
                                    [153, 151, 124],
                                    [126, 133, 113]],
                          'false': [125, 145, 135, 173, 128],
                          'val': [0.0486, 0.0474, 0.0477, 0.0472, 0.0467]},
           'curves': {'train': [1.9988, 0.9618, 0.5964, 0.4471, 0.307, 0.2018, 0.1156, 0.0834],
                      'val': [1.6442, 1.2311, 0.6653, 0.7832, 0.4969, 0.4002, 0.3803, 0.3838]}}


# --- the course's versions ------------------------------------------------------------------------------------------

def focal_loss(logits, classes, gamma: float = GAMMA):
    """The focal loss, as the learner's: for every anchor that is not ignored, its cross-entropy, p_t = e^-ce, and
    (1 - p_t)^gamma times the cross-entropy; summed, and divided by the number of anchors that learn a shape."""
    kept = classes >= 0
    ce = F.cross_entropy(logits[kept], classes[kept], reduction="none")
    p = torch.exp(-ce)
    return ((1 - p) ** gamma * ce).sum() / (classes > 0).sum()


def mean_focal_loss(logits, classes, gamma: float = GAMMA):
    """The focal loss divided by every anchor that is not ignored, as the one-stage detector divided its cross-entropy:
    for the comparison."""
    kept = classes >= 0
    ce = F.cross_entropy(logits[kept], classes[kept], reduction="none")
    p = torch.exp(-ce)
    return ((1 - p) ** gamma * ce).mean()


def background_bias(pi: float = PI, kinds: int = KINDS) -> float:
    """The bias b of the background logit, as the learner's: softmax over (b, 0, ..., 0) gives background 1 - pi, so
    e^b / (e^b + kinds) = 1 - pi."""
    return math.log(kinds * (1 - pi) / pi)


def start_as_background(model, pi: float = PI, bias_fn=background_bias):
    """The head's biases so that every anchor starts at background with a probability of about 1 - pi: for each anchor
    of a square, the bias of its background logit bias_fn(pi, KINDS), of its other logits 0. Its offsets keep theirs."""
    with torch.no_grad():
        b = model.head.bias.view(-1, 8)
        b[:, :4] = 0.0
        b[:, 0] = bias_fn(pi, KINDS)


def box_loss(out, classes, t):
    """The mean absolute difference of the offsets over the anchors that learn a shape, as before."""
    obj = classes > 0
    return (out[:, :, 4:][obj] - t[obj]).abs().mean()


def detector_loss(gamma: float = GAMMA, focal_fn=focal_loss):
    """The loss of the detector: the focal loss of the logits plus the box loss."""
    def loss(out, classes, t):
        return focal_fn(out[:, :, :4], classes, gamma) + box_loss(out, classes, t)
    return loss


def loss_of(variant: str):
    gamma, _, divide = LOSSES[variant]
    if variant in ("mean", "mean-start"):
        return _ONE.detection_loss
    return detector_loss(gamma, focal_loss if divide == "shapes" else mean_focal_loss)


def train_course(variant: str = "focal", seed: int = 0, heard=None) -> tuple:
    """A detector of the course, trained with one loss of LOSSES and `seed` on one thread, as a check runs: (model,
    history)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = _FPN.make("fpn")
        if LOSSES[variant][1]:
            start_as_background(model)
        history = _FPN.train(model, "fpn", seed, heard, loss_of(variant))
    finally:
        torch.set_num_threads(threads)
    return model, history


def where(out) -> dict:
    """Where the loss of a detector comes from on the test scenes: how many anchors are easy background (a probability
    of background above EASY), other background, and anchors that learn a shape; and the share of each in the summed
    cross-entropy and in the summed focal loss with gamma 2."""
    classes = course_targets()[1][0]
    kept = classes >= 0
    ce = F.cross_entropy(out[:, :, :4][kept], classes[kept], reduction="none")
    p = torch.exp(-ce)
    bg = classes[kept] == 0
    groups = (bg & (p > EASY), bg & (p <= EASY), ~bg)
    shares = {}
    for name, loss in (("ce", ce), ("fl", (1 - p) ** GAMMA * ce)):
        total = float(loss.sum())
        shares[name] = [round(float(loss[g].sum()) / total, 4) for g in groups]
    # how sure it is of the anchors that learn a shape: their mean p_t, and the share of them above 0.5
    sure = [round(float(p[~bg].mean()), 4), round(float((p[~bg] > 0.5).float().mean()), 4)]
    return {"anchors": [int(g.sum()) for g in groups]} | shares | {"shapes_p": sure}


def measure_variant(variant: str, seed: int) -> dict:
    """Train one detector and measure what TRAINED holds for it."""
    model, history = train_course(variant, seed)
    ss, Xt = scene_set("test")
    o = outputs(model, Xt)
    out = {k: v for k, v in summary(o, ss, anchors()).items() if k not in ("levels", "shapes")}
    out["val"] = round(history[-1]["val_loss"], 4)
    if seed == 0 and variant in BY_GAMMA.values():
        out["scene"] = shown(detect(o[TEST.index(SEED)], anchors()))
        out["where"] = where(o)
        if variant == "focal":
            out["curves"] = {"train": [round(h["train_loss"], 4) for h in history],
                             "val": [round(h["val_loss"], 4) for h in history]}
    return out


def measure_trained(seeds=range(5)) -> dict:
    """Everything TRAINED holds, measured again: every loss of LOSSES with every seed. Takes two to three minutes per
    training on one thread, about two hours in all."""
    out = {}
    for variant in LOSSES:
        runs = [measure_variant(variant, seed) for seed in seeds]
        out[variant] = {key: [r[key] for r in runs] for key in ("map", "aps", "map75", "found", "false", "val")}
        for key in ("scene", "where"):
            if key in runs[0]:
                out[variant][key] = runs[0][key]
        if "curves" in runs[0]:
            out["curves"] = runs[0]["curves"]
    return out


def hard() -> dict:
    """cv-fpn's FPN detector, trained with hard negative mining, over the seeds 0 to 4."""
    return _FPN.TRAINED["fpn"]


# --- the worked example ---------------------------------------------------------------------------------------------

EXAMPLE_LOGITS = [[6.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0],
                  [0.0, 5.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]
EXAMPLE_CLASSES = [0, 0, 0, -1, 1, 2]


def worked() -> dict:
    """The worked example's numbers, as its cell computes them: the cross-entropy, p_t and focal loss of six anchors
    and the loss they give; then the scene of the worked example at the start of training, every logit 0: how many
    anchors are background, ignored and shapes, and the share of the background in the summed cross-entropy and the
    summed focal loss."""
    logits, classes = torch.tensor(EXAMPLE_LOGITS), torch.tensor(EXAMPLE_CLASSES)
    kept = classes >= 0
    ce = F.cross_entropy(logits[kept], classes[kept], reduction="none")
    p = torch.exp(-ce)
    fl = (1 - p) ** 2 * ce
    n = (classes > 0).sum()
    s = example()
    label = torch.tensor(scenes.match_anchors(anchors().numpy(), [shape.box for shape in s.shapes]))
    scene_classes = torch.zeros(len(label), dtype=torch.int64)
    scene_classes[label == -2] = -1
    for i, shape in enumerate(s.shapes):
        scene_classes[label == i] = shape.label + 1
    start = torch.zeros(len(label), 4)
    kept_s = scene_classes >= 0
    ce_s = F.cross_entropy(start[kept_s], scene_classes[kept_s], reduction="none")
    fl_s = (1 - torch.exp(-ce_s)) ** 2 * ce_s
    bg = scene_classes[kept_s] == 0
    return {"ce": [round(v, 4) for v in ce.tolist()], "p": [round(v, 4) for v in p.tolist()],
            "fl": [float(f"{v:.5g}") for v in fl.tolist()], "ratio": [float(f"{v:.5g}") for v in (ce / fl).tolist()],
            "loss": [round(float(ce.sum() / n), 4), round(float(fl.sum() / n), 4)],
            "scene": {"background": int(bg.sum()), "ignored": int((scene_classes == -1).sum()),
                      "shapes": int((scene_classes > 0).sum()), "ce": round(float(ce_s[0]), 4), "fl": round(float(fl_s[0]), 4),
                      "ce_share": round(float(ce_s[bg].sum() / ce_s.sum()), 4),
                      "fl_share": round(float(fl_s[bg].sum() / fl_s.sum()), 4)},
            "bias": round(background_bias(), 4)}


def measure() -> dict:
    return worked()


# --- pictures ---------------------------------------------------------------------------------------------------------

def focal(p: float, gamma: float) -> float:
    """The focal loss of one anchor whose probability of what it learns is p."""
    return (1 - p) ** gamma * -math.log(p)


_PS = [i / 200 for i in range(2, 200)] + [0.999]   # p from 0.01 to 0.999
_STYLE = {0.0: ("ink-3", 2.0, True), 0.5: ("loss", 1.3, True), 1.0: ("loss", 1.3, False), 2.0: ("loss", 3.0, False),
          5.0: ("loss", 1.3, True)}
TOP = 4.0                     # the highest loss the curves show


def _curves_panel(words: dict, chosen: float | None = None, gammas=DRAWN) -> chart.Panel:
    """The focal loss over p for every gamma in `gammas`; with `chosen`, that one thick and the others thin and grey."""
    p = chart.Panel((0, 1), (0, TOP), title=words["title"], xlabel=words["x"], ylabel=words["y"],
                    xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 1, 2, 3, 4])
    for g in gammas:
        role, width, dashed = _STYLE[g] if chosen is None else (("loss", 3.0, False) if g == chosen else ("ink-3", 1.3, False))
        ps = [x for x in _PS if focal(x, g) <= TOP * 1.05]   # the part of the curve inside the chart
        p.line(ps, [focal(x, g) for x in ps], role, width, dashed)
    return p


def focal_picture(words: dict) -> str:
    """The focal loss over p_t for gamma 0 (the cross-entropy), 0.5, 1, 2 and 5, as in the paper's first figure, with a
    key in the empty upper part."""
    p = _curves_panel(words)
    for i, g in enumerate(DRAWN):   # the key: a short line of each curve's look, and its gamma, where every curve is below 2
        y = TOP * (0.95 - 0.1 * i)
        role, width, dashed = _STYLE[g]
        p.line([0.15, 0.25], [y, y], role, width, dashed)
        p.text(0.28, y, words["key"][i], "ink", "start")
    svg = chart.figure([p], alt=words["alt"], width=300, height=240)
    return chart.with_decimal_comma(svg) if words.get("point") == "," else svg


def gamma_frame(g: float, words: dict) -> tuple[str, dict]:
    """For the slider: the curves with this gamma thick, and next to it, the scene of the worked example with the boxes
    of 0.5 or more of the course's detector trained with this gamma (seed 0)."""
    g = float(g)
    r = TRAINED[BY_GAMMA[g]]
    boxes = r["scene"]
    plot = chart.figure([_curves_panel(words["chart"], chosen=g, gammas=GAMMAS)], alt=words["chart"]["alt"].format(g=_g(g)),
                        width=276, height=236)
    if words["point"] == ",":
        plot = chart.with_decimal_comma(plot)
    title = words["title"][0 if len(boxes) == 1 else 1]
    scene = _FPN._scene_svg(example(), boxes, title.format(g=_g(g), n=len(boxes)), names=words, figsize=(2.9, 3.0), size=9)
    stop = {"factor": round(1 / (1 - EASY) ** g), "found": sum(r["found"][0]), "false": r["false"][0], "map": r["map"][0],
            "boxes": len(boxes)}
    return _ONE._side_by_side([(plot, 276), (scene, 250)], words["alt"].format(g=_g(g), n=len(boxes))), stop


def _g(g: float) -> str:
    return f"{g:g}"


def comparison_table(words: dict, lang: str, rows=("mean", "hard", "gamma0", "focal", "no-start")) -> str:
    """The losses side by side, as a Markdown table: for each, the shapes of the test scenes it finds of 786, its mAP
    at 0.5 and 0.75, and its false boxes, from the lowest to the highest over the seeds 0 to 4."""
    def span(values, show):
        low, high = min(values), max(values)
        return show(low) if low == high else f"{show(low)}–{show(high)}"

    lines = ["| " + " | ".join(words["header"]) + " |", "|" + "---|" * len(words["header"])]
    for variant in rows:
        r = hard() if variant == "hard" else TRAINED[variant]
        cells = [words["names"][variant], span([sum(f) for f in r["found"]], str),
                 span(r["map"], lambda x: number(x, 3, lang)), span(r["map75"], lambda x: number(x, 3, lang)),
                 span(r["false"], str)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    return _ONE.curves_picture(curves, labels, lang, at)


def rewind_picture(boxes: list, words: dict) -> str:
    return _FPN.rewind_picture(boxes, words)


# --- exercise 2: focal_loss -------------------------------------------------------------------------------------------

_DOTS = _ONE._DOTS
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close

_L_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef focal_loss(logits, classes, gamma):\n"
           "    # logits: the 4 logits of every anchor, background first, shape (N, A, 4)\n"
           "    # classes: what every anchor learns, shape (N, A): 0 background, 1 to 3 a kind, -1 ignored\n"
           "    # give back the focal loss of every anchor that is not ignored, summed, and divided\n"
           "    # by the number of anchors that learn a shape\n")
_L_START = _L_HEAD + "    ...\n"
_L_KEPT = "    kept = classes >= 0\n"
_L_CE = "    ce = F.cross_entropy(logits[kept], classes[kept], reduction=\"none\")\n"
_L_P = "    p = torch.exp(-ce)\n"
_L_RETURN = "    return ((1 - p) ** gamma * ce).sum() / (classes > 0).sum()\n"


def _l(kept=_L_KEPT, ce=_L_CE, p=_L_P, ret=_L_RETURN) -> str:
    return _L_HEAD + kept + ce + p + ret


_L = _l()
_L_SCENARIOS = [
    # the six anchors of the worked example, as one scene
    "import torch\n\nlogits = torch.tensor([[" + ", ".join(str(row) for row in EXAMPLE_LOGITS) + "]])\n"
    "classes = torch.tensor([" + str(EXAMPLE_CLASSES) + "])\n"
    "[float(focal_loss(logits, classes, 2.0)), float(focal_loss(logits, classes, 0.0))]",
    # two scenes of 40 anchors with random logits and classes, and five values of gamma
    "import torch\n\ntorch.manual_seed(1)\nlogits = torch.randn(2, 40, 4) * 3\nclasses = torch.randint(-1, 4, (2, 40))\n"
    "[float(focal_loss(logits, classes, g)) for g in (0.0, 0.5, 1.0, 2.0, 5.0)]",
    # three scenes of 2,720 anchors, as many as the detector has: 12 learn a shape in each, 8 are ignored
    "import torch\n\ntorch.manual_seed(2)\nlogits = torch.randn(3, 2720, 4) * 4\n"
    "classes = torch.zeros(3, 2720, dtype=torch.int64)\nclasses[:, :12] = torch.randint(1, 4, (3, 12))\n"
    "classes[:, 12:20] = -1\n[float(focal_loss(logits, classes, g)) for g in (0.0, 2.0)]",
    # as in training: the loss followed back to the logits
    "import torch\n\ntorch.manual_seed(3)\nlogits = torch.randn(2, 40, 4, requires_grad=True)\nclasses = torch.randint(-1, 4, (2, 40))\n"
    "loss = focal_loss(logits, classes, 2.0)\nloss.backward()\n[float(loss), float(logits.grad.abs().sum())]",
]
_L_WRONG = {
    "mean": _l(ret="    return ((1 - p) ** gamma * ce).mean()\n"),
    "not-divided": _l(ret="    return ((1 - p) ** gamma * ce).sum()\n"),
    "sure-weight": _l(ret="    return (p ** gamma * ce).sum() / (classes > 0).sum()\n"),
    "no-ce": _l(ret="    return ((1 - p) ** gamma).sum() / (classes > 0).sum()\n"),
    "exp-sign": _l(p="    p = torch.exp(ce)\n"),
    "one-number": _l(ce="    ce = F.cross_entropy(logits[kept], classes[kept])\n"),
    "shapes-only": _l(kept="    kept = classes > 0\n"),
}


def _target_out_of_bounds(e: BaseException) -> bool:
    return isinstance(e, (IndexError, RuntimeError)) and re.search(r"Target -?\d+ is out of bounds", str(e)) is not None


def _no_way_back(e: BaseException) -> bool:
    """loss.backward() on a Python number, or on a tensor that training cannot follow back."""
    return (isinstance(e, AttributeError) and "backward" in str(e)) or \
        (isinstance(e, RuntimeError) and "does not require grad" in str(e))


def _rows_expected(e: BaseException) -> bool:
    """F.cross_entropy got the logits of whole scenes, (N, A, 4), instead of rows of 4."""
    return isinstance(e, (ValueError, RuntimeError)) and re.search(r"Expected target size|expected target size|only batches of spatial targets", str(e)) is not None


_L_MISTAKES = [
    Mistake("ignored", error=_target_out_of_bounds,
            en="Python stops in `F.cross_entropy`: an anchor that is ignored has the class −1, which is not one of the 4 classes. First keep the anchors that are not ignored, `kept = classes >= 0`, and give `F.cross_entropy` only theirs: `logits[kept]` and `classes[kept]`.",
            nl="Python stopt in `F.cross_entropy`: een anchor die genegeerd wordt heeft de klasse −1, en dat is geen van de 4 klassen. Houd eerst de anchors over die niet genegeerd worden, `kept = classes >= 0`, en geef `F.cross_entropy` alleen die van hen: `logits[kept]` en `classes[kept]`."),
    Mistake("rows", error=_rows_expected,
            en="Python stops in `F.cross_entropy`: it wants one row of 4 logits per anchor, and gets whole scenes, of shape (N, A, 4). `logits[kept]` gives the rows of the anchors that are not ignored, one under the other, and `classes[kept]` their classes.",
            nl="Python stopt in `F.cross_entropy`: die wil één rij van 4 logits per anchor, en krijgt hele scènes, van shape (N, A, 4). `logits[kept]` geeft de rijen van de anchors die niet genegeerd worden, onder elkaar, en `classes[kept]` hun klassen."),
    Mistake("number", error=_no_way_back,
            en="Python stops at `loss.backward()`, which training runs after every batch: your `focal_loss` gives back a number that training cannot follow back to the weights. Give back the tensor itself, without `.item()`, `float(...)` or `.detach()`.",
            nl="Python stopt bij `loss.backward()`, wat het trainen na elke batch doet: je `focal_loss` geeft een getal terug dat het trainen niet kan terugvolgen naar de weights. Geef de tensor zelf terug, zonder `.item()`, `float(...)` of `.detach()`."),
    Mistake("mean", wrong=_L_WRONG["mean"],
            en=_S_EN + ": you divide by every anchor that is not ignored, about 2,700 in a scene, as the mean of your one-stage detector did. RetinaNet divides the sum by the anchors that learn a shape, about 10 in a scene: `(classes > 0).sum()`. The easy background anchors add almost nothing to the focal loss, so they should not make it smaller either.",
            nl=_S_NL + ": je deelt door elke anchor die niet genegeerd wordt, zo'n 2.700 in een scène, zoals het gemiddelde van je one-stage detector deed. RetinaNet deelt de som door de anchors die een vorm leren, zo'n 10 in een scène: `(classes > 0).sum()`. De makkelijke achtergrond-anchors voegen bijna niets toe aan de focal loss, dus ze moeten hem ook niet kleiner maken."),
    Mistake("not-divided", wrong=_L_WRONG["not-divided"],
            en=_S_EN + ": you give back the sum. Divide it by the number of anchors that learn a shape, `(classes > 0).sum()`, so that a batch of 32 scenes gives a loss of the same size as a batch of 2.",
            nl=_S_NL + ": je geeft de som terug. Deel die door het aantal anchors dat een vorm leert, `(classes > 0).sum()`, zodat een batch van 32 scènes een loss van dezelfde grootte geeft als een batch van 2."),
    Mistake("sure-weight", wrong=_L_WRONG["sure-weight"],
            en=_S_EN + ": you weight every anchor by `p ** gamma`, so the anchors the detector is surest of count most. The focal loss does the opposite: `(1 - p) ** gamma`, small when p_t is close to 1.",
            nl=_S_NL + ": je weegt elke anchor met `p ** gamma`, dus de anchors waar de detector het zekerst van is tellen het zwaarst. De focal loss doet het omgekeerde: `(1 - p) ** gamma`, klein als p_t dicht bij 1 ligt."),
    Mistake("no-ce", wrong=_L_WRONG["no-ce"],
            en=_S_EN + ": `(1 - p) ** gamma` is only the weight of every anchor. The focal loss is that weight times the cross-entropy: `(1 - p) ** gamma * ce`.",
            nl=_S_NL + ": `(1 - p) ** gamma` is alleen het gewicht van elke anchor. De focal loss is dat gewicht maal de cross-entropy: `(1 - p) ** gamma * ce`."),
    # with a gamma of 0.5, a negative number to the power 0.5 gives nan: known by its text, not its outcome
    Mistake("exp-sign", matches=lambda got, text: re.search(r"torch\.exp\(\s*ce\s*\)", text) is not None,
            en=_S_EN + ": `torch.exp(ce)` is larger than 1, and a probability is not. The cross-entropy is −log p_t, so p_t is `torch.exp(-ce)`, with a minus.",
            nl=_S_NL + ": `torch.exp(ce)` is groter dan 1, en een kans niet. De cross-entropy is −log p_t, dus p_t is `torch.exp(-ce)`, met een min."),
    Mistake("one-number", wrong=_L_WRONG["one-number"],
            en=_S_EN + ": without `reduction=\"none\"`, `F.cross_entropy` gives one number, the mean over all anchors, and every anchor gets the same weight. The focal loss weights every anchor by its own p_t: `reduction=\"none\"` gives a cross-entropy for every anchor.",
            nl=_S_NL + ": zonder `reduction=\"none\"` geeft `F.cross_entropy` één getal, het gemiddelde over alle anchors, en krijgt elke anchor hetzelfde gewicht. De focal loss weegt elke anchor met zijn eigen p_t: `reduction=\"none\"` geeft een cross-entropy voor elke anchor."),
    Mistake("shapes-only", wrong=_L_WRONG["shapes-only"],
            en=_S_EN + ": you only count the anchors that learn a shape, so the detector never learns background, and gives boxes everywhere. The focal loss counts every anchor that is not ignored, `classes >= 0`, background too; it only divides by the anchors that learn a shape.",
            nl=_S_NL + ": je telt alleen de anchors die een vorm leren, dus de detector leert nooit achtergrond, en geeft overal boxes. De focal loss telt elke anchor die niet genegeerd wordt, `classes >= 0`, ook achtergrond; hij deelt alleen door de anchors die een vorm leren."),
]

# --- exercise 3: background_bias --------------------------------------------------------------------------------------

_B_HEAD = ("import math\n\n\ndef background_bias(pi, kinds):\n"
           "    # pi: the probability of a shape that every anchor starts with, 0.01 in RetinaNet\n"
           "    # kinds: how many kinds there are besides background, 3 in the course's scenes\n"
           "    # give back the bias b of the background logit for which softmax over (b, 0, ..., 0),\n"
           "    # with a 0 for every kind, gives background a probability of 1 - pi\n")
_B_START = _B_HEAD + "    ...\n"
_B = _B_HEAD + "    return math.log(kinds * (1 - pi) / pi)\n"
_B_SCENARIOS = [
    # RetinaNet's pi and the course's 3 kinds
    "background_bias(0.01, 3)",
    # other values of pi and other numbers of kinds
    "[background_bias(pi, k) for pi, k in [(0.1, 1), (0.5, 3), (0.25, 80), (0.01, 1)]]",
    # softmax over the bias and a 0 for each of 4 kinds: background gets 1 - pi
    "import torch\n\ne = torch.exp(torch.tensor([background_bias(0.05, 4), 0.0, 0.0, 0.0, 0.0]))\n(e / e.sum()).tolist()",
]
_B_WRONG = {
    "one-kind": _B_HEAD + "    return math.log((1 - pi) / pi)\n",
    "upside-down": _B_HEAD + "    return math.log(pi / (kinds * (1 - pi)))\n",
    "no-log": _B_HEAD + "    return kinds * (1 - pi) / pi\n",
    "paper": _B_HEAD + "    return -math.log((1 - pi) / pi)\n",
}
_B_MISTAKES = [
    Mistake("one-kind", wrong=_B_WRONG["one-kind"],
            en=_S_EN + ": that is the bias for one kind. With softmax over background and `kinds` kinds, every kind's logit of 0 adds e⁰ = 1 under the line: e^b / (e^b + kinds) = 1 − pi. Solve that for b.",
            nl=_S_NL + ": dat is de bias voor één soort. Met softmax over achtergrond en `kinds` soorten voegt de logit 0 van elke soort e⁰ = 1 toe onder de streep: e^b / (e^b + kinds) = 1 − pi. Los dat op naar b."),
    Mistake("upside-down", wrong=_B_WRONG["upside-down"],
            en=_S_EN + ": your bias is negative, so softmax gives background less than the kinds. From e^b / (e^b + kinds) = 1 − pi follows e^b · pi = kinds · (1 − pi): the fraction is the other way around.",
            nl=_S_NL + ": je bias is negatief, dus softmax geeft achtergrond minder dan de soorten. Uit e^b / (e^b + kinds) = 1 − pi volgt e^b · pi = kinds · (1 − pi): de breuk is andersom."),
    Mistake("no-log", wrong=_B_WRONG["no-log"],
            en=_S_EN + ": that is e^b, not b. The bias itself is its logarithm: `math.log(...)`.",
            nl=_S_NL + ": dat is e^b, niet b. De bias zelf is de logaritme daarvan: `math.log(...)`."),
    Mistake("paper", wrong=_B_WRONG["paper"],
            en=_S_EN + ": that is the bias of the paper, which every kind's logit gets with a sigmoid of its own, and it is negative. Here softmax runs over background and the kinds, and only the background logit gets a bias, a positive one: solve e^b / (e^b + kinds) = 1 − pi for b.",
            nl=_S_NL + ": dat is de bias van de paper, die de logit van elke soort krijgt met een eigen sigmoid, en hij is negatief. Hier loopt softmax over achtergrond en de soorten, en krijgt alleen de achtergrondlogit een bias, een positieve: los e^b / (e^b + kinds) = 1 − pi op naar b."),
]

EXERCISES = {
    "focal-loss": ScenarioBox(
        name="focal-loss",
        start=_L_START,
        solution=_L,
        scenarios=_L_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "focal"),
        first=[_DOTS],
        mistakes=_L_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: which anchors are not ignored; the cross-entropy of each of them, one number per anchor; p_t from it; the focal loss of every anchor; and their sum, divided by the number of anchors that learn a shape.",
                      "nl": "Vervang de `...` door je code: welke anchors niet genegeerd worden; de cross-entropy van elk van hen, één getal per anchor; p_t daaruit; de focal loss van elke anchor; en hun som, gedeeld door het aantal anchors dat een vorm leert."},
            "wrong": {"en": _S_EN + ". Does every anchor that is not ignored get its own cross-entropy? Is p_t `torch.exp(-ce)`, and the focal loss `(1 - p) ** gamma * ce`? And do you divide the sum by the number of anchors that learn a shape?",
                      "nl": _S_NL + ". Krijgt elke anchor die niet genegeerd wordt zijn eigen cross-entropy? Is p_t `torch.exp(-ce)`, en de focal loss `(1 - p) ** gamma * ce`? En deel je de som door het aantal anchors dat een vorm leert?"},
            "almost": {"en": "Right for the six anchors. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of scenes and anchors, and any gamma?",
                       "nl": "Goed voor de zes anchors. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal scènes en anchors, en elke gamma?"},
            "correct": {"en": "Your `focal_loss` is right: every anchor counts, each as much as the detector is still wrong about it. It is now in your project, as `{saved}`.",
                        "nl": "Je `focal_loss` klopt: elke anchor telt mee, elk zoveel als de detector er nog naast zit. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `focal_loss` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `focal_loss` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "background-bias": ScenarioBox(
        name="background-bias",
        start=_B_START,
        solution=_B,
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "background"),
        first=[_DOTS],
        mistakes=_B_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the bias b for which e^b, divided by e^b plus 1 for every kind, is 1 − pi; solve that for b.",
                      "nl": "Vervang de `...` door je code: de bias b waarvoor e^b, gedeeld door e^b plus 1 voor elke soort, gelijk is aan 1 − pi; los dat op naar b."},
            "wrong": {"en": _S_EN + ". Softmax over (b, 0, ..., 0) gives background e^b / (e^b + kinds). Is that 1 − pi with your b?",
                      "nl": _S_NL + ". Softmax over (b, 0, ..., 0) geeft achtergrond e^b / (e^b + kinds). Is dat 1 − pi met jouw b?"},
            "almost": {"en": "Right for 0.01 and 3 kinds. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any pi and any number of kinds?",
                       "nl": "Goed voor 0,01 en 3 soorten. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elke pi en elk aantal soorten?"},
            "correct": {"en": "Your `background_bias` is right: with it, every anchor starts at background with a probability of 1 − pi. It is now in your project, as `{saved}`.",
                        "nl": "Je `background_bias` klopt: daarmee begint elke anchor bij achtergrond met een kans van 1 − pi. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `background_bias` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `background_bias` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "easy" and "versus" are choices, "why-gamma" the open question.
REQUIRED = ["easy", "focal-loss", "background-bias", "versus", "why-gamma"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("focal-loss", _L_START, "empty", None),
    ("focal-loss", _L, "correct", None),
    # p_t and the focal loss in one line
    ("focal-loss", _l(p="", ret="    return ((1 - torch.exp(-ce)) ** gamma * ce).sum() / (classes > 0).sum()\n"), "correct", None),
    # the anchors that learn a shape counted with a mask of their own
    ("focal-loss", _l(ret="    shapes = classes > 0\n    return ((1 - p) ** gamma * ce).sum() / shapes.sum()\n"), "correct", None),
    # without the mask: every anchor, the ignored ones too
    ("focal-loss", _L_HEAD + "    ce = F.cross_entropy(logits.reshape(-1, 4), classes.reshape(-1), reduction=\"none\")\n" + _L_P + _L_RETURN,
     "error", "ignored"),
    ("focal-loss", _L_HEAD + "    ce = F.cross_entropy(logits, classes, reduction=\"none\")\n" + _L_P + _L_RETURN, "error", "rows"),
    ("focal-loss", _l(ret="    ...\n"), "wrong", "dots-left"),
    # a Python number, or a tensor cut off from training
    ("focal-loss", _l(ret="    return float(((1 - p) ** gamma * ce).sum() / (classes > 0).sum())\n"), "error", "number"),
    ("focal-loss", _l(ret="    return ((1 - p) ** gamma * ce).sum().item() / (classes > 0).sum().item()\n"), "error", "number"),
    ("focal-loss", _l(ret="    return (((1 - p) ** gamma * ce).sum() / (classes > 0).sum()).detach()\n"), "error", "number"),
    # divided by every anchor that is not ignored, as a sum
    ("focal-loss", _l(ret="    return ((1 - p) ** gamma * ce).sum() / kept.sum()\n"), "wrong", "mean"),
] + [("focal-loss", code, "wrong", key) for key, code in _L_WRONG.items()] + [
    ("background-bias", _B_START, "empty", None),
    ("background-bias", _B, "correct", None),
    # the logarithms of the three parts
    ("background-bias", _B_HEAD + "    return math.log(kinds) + math.log(1 - pi) - math.log(pi)\n", "correct", None),
    # the dots left in one line of two
    ("background-bias", _B_HEAD + "    e_b = ...\n    return math.log(e_b)\n", "wrong", "dots-left"),
    # the course's 3 kinds as a fixed number: right for 3, wrong for the others
    ("background-bias", _B_HEAD + "    return math.log(3 * (1 - pi) / pi)\n", "almost", None),
] + [("background-bias", code, "wrong", key) for key, code in _B_WRONG.items()]


# --- the demo: the course's FPN detector with the learner's focal loss ------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


_Stopped, _named = _FPN._Stopped, _FPN._named


def _demo(show=None):
    """For the demo: None when focal.py or background.py is not in the project yet; "background" when the learner's
    background_bias stops or gives no number; "focal" when their focal_loss stops or does not give one number that
    training can follow back, and when it stops while training; "pyramid", "head_rows" or "offsets" when that module
    stops; "slow" when the training takes too long; else a dict with the loss after every epoch ("curves"), the boxes
    on the SHOWN scenes after every epoch ("moments"), the summary of the trained detector on the test scenes, the bias
    it started with, and the modules the course's versions stood in for. `show` hears the run while it trains."""
    if not (_saved("focal") and _saved("background")):
        return None
    names = {}
    for module, name, key in (("focal", "focal_loss", "focal"), ("background", "background_bias", "background"),
                              ("pyramid", "pyramid", "pyramid"), ("head_rows", "head_rows", "head_rows"),
                              ("offsets", "boxes_from", "offsets")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    try:
        bias = float(names["background_bias"](PI, KINDS))
    except Exception:
        return "background"
    if not math.isfinite(bias):
        return "background"
    (C, _), _ = course_targets()
    _, X = scene_set("train")
    torch.manual_seed(0)
    net = _FPN.FPNDetector()
    want = [(1, _FPN.D, cells, cells) for _, cells, _ in _FPN.LEVELS]
    try:   # one scene through the learner's pyramid and head_rows: the shapes the training needs
        cs = net.backbone(X[:1].float() / 255)
        maps = names["pyramid"](cs, net.lateral, net.smooth)
        if not (len(maps) == 4 and all(torch.is_tensor(p) and tuple(p.shape) == w and p.requires_grad for p, w in zip(maps, want))):
            return "pyramid"
    except Exception:
        return "pyramid"
    try:
        rows = names["head_rows"](net.head, _FPN.pyramid(cs, net.lateral, net.smooth))
        if not (torch.is_tensor(rows) and tuple(rows.shape) == (1, len(anchors()), 8) and rows.requires_grad):
            return "head_rows"
    except Exception:
        return "head_rows"
    try:   # two scenes through the learner's focal loss: one number that training can follow back
        out = net(X[:2].float() / 255)
        value = names["focal_loss"](out[:, :, :4], C[:2], GAMMA)
        if not (torch.is_tensor(value) and value.dim() == 0 and value.requires_grad and torch.isfinite(value)):
            return "focal"
    except Exception:
        return "focal"
    focal_of = _named(names["focal_loss"], "focal")
    pyr, rows_of, boxes_of = (_named(names["pyramid"], "pyramid"), _named(names["head_rows"], "head_rows"),
                              _named(names["boxes_from"], "offsets"))
    torch.manual_seed(0)   # the same start as the course's run
    model = _FPN.FPNDetector(pyr, rows_of)
    start_as_background(model, PI, lambda pi, kinds: bias)
    a = anchors()
    moments, broken = [], []

    def heard(p):
        if "out" in p:
            try:
                moments.append([shown(detect(p["out"][TEST.index(seed)], a, boxes_fn=boxes_of)) for seed in SHOWN])
            except _Stopped as e:   # the learner's boxes_from stopped (any other error is the course's own)
                broken.append(e.key)
                moments.append([[] for _ in SHOWN])
            p = {k: v for k, v in p.items() if k != "out"}
        if show is not None:
            try:
                show(p)
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass

    try:   # with_time_limit trains on one thread, as the course's run
        history = with_time_limit(lambda: _FPN.train(model, "fpn", 0, heard, detector_loss(GAMMA, focal_of)), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if broken:
        return broken[0]
    if not all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history):
        return "focal"
    test_ss, Xt = scene_set("test")
    try:
        result = summary(outputs(model, Xt), test_ss, a, boxes_fn=boxes_of)
    except _Stopped as e:   # the learner's module that stopped; any other error is the course's own
        return e.key
    result.pop("levels", None)
    curves = {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]}
    return result | {"curves": curves, "moments": moments, "bias": bias, "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("focal", "background", "pyramid", "head_rows", "offsets"):
        path = projects.folder() / "cnn" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]
