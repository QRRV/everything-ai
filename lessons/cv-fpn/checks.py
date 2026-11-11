# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-fpn: pyramid, which makes the maps of a feature pyramid network from the maps of the
backbone (from the top down: a 1×1 convolution of every map, the level above made twice as large with
nearest-neighbor upsampling and added, then a 3×3 convolution), and head_rows, which gives the anchors of every level
their rows with one head. Both are saved in the learner's project (work/projects/cnn/pyramid.py and head_rows.py)
when right. The demo trains the course's FPN detector with them on the course's scenes with small and large shapes,
finds the shapes of new scenes with the learner's boxes_from and NMS, and counts what it finds of every size. And the
pictures: the pyramid itself, and for the slider, the boxes that each level of the course's trained detector finds on
the scene of the worked example. Everything is computed from the course's own scenes (scenes.py): no download."""
import functools
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch import nn

from everything_ai import chart, paths, projects, scenes
from everything_ai.check import Mistake, ScenarioBox, Timeout, run_code, with_time_limit
from everything_ai import yours


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_fpn",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The targets of every anchor, the training, the mAP and the pictures of boxes on scenes
_ONE = _lesson_checks("cv-one-stage-detector")

# The demo finds the boxes with the learner's boxes_from of cv-anchors; where it is missing or broken, the course's
# version stands in, and the lesson says so.
BUILDS_ON = (("cnn", "offsets", "cv-anchors", "boxes-from"),)
projects.builds_on(*BUILDS_ON)

SEED = 26                     # the scene of the worked example: scenes.scene(26, spread=True), a test scene
KINDS = _ONE.KINDS
N_TRAIN, N_TEST, TRAIN, TEST = _ONE.N_TRAIN, _ONE.N_TEST, _ONE.TRAIN, _ONE.TEST   # the seeds, now with spread=True
EPOCHS, LR, BATCH = _ONE.EPOCHS, _ONE.LR, _ONE.BATCH
SCORE, NMS_IOU, LOW, AP_IOU = _ONE.SCORE, _ONE.NMS_IOU, _ONE.LOW, _ONE.AP_IOU
iou, mean_ap, hits, lr_factor = _ONE.iou, _ONE.mean_ap, _ONE.hits, _ONE.lr_factor
offsets, boxes_from, targets = _ONE.offsets, _ONE.boxes_from, _ONE.targets
boxes_and_labels, outputs, scored = _ONE.boxes_and_labels, _ONE.outputs, _ONE.scored
number, whole, low_high = _ONE.number, _ONE.whole, _ONE.low_high

D = 32                        # the channels of every level of the pyramid
LEVELS = ((2, 32, 4), (3, 16, 8), (4, 8, 16), (5, 4, 32))   # level k: its squares (32 by 32 to 4 by 4) of 2**k pixels
CHANNELS = (32, 64, 128, 128)  # the channels of the backbone's maps C2 to C5
HARD = 3                      # background anchors the loss keeps per anchor that learns a shape, the hardest, as SSD
SMALL, LARGE = 16, 32         # a shape is small below a box of 16 x 16 pixels in area, large from 32 x 32 on
VARIANTS = ("fpn", "bottom-up", "deep", "sharp")   # the detectors the course compares, each with its 5 seeds
SHOWN = (26, 23, 11)          # the test scenes of the rewind
TRAIN_LIMIT = 600             # seconds the demo gives the training: the note says ten minutes
SHOWN_SHAPES = sum(len(scenes.scene(seed, spread=True).shapes) for seed in SHOWN)

# The numbers of the worked example, measured without training. `measure()` measures them again; the test of the
# lesson checks that both are the same. "shapes": for every shape of the scene, its kind, its size (size_of), and the
# level of every anchor that learns it, in the order of the anchors.
REFERENCE = {'c': [(1, 32, 32, 32), (1, 64, 16, 16), (1, 128, 8, 8), (1, 128, 4, 4)],
             'up': [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]],
             'p': [(1, 32, 32, 32), (1, 32, 16, 16), (1, 32, 8, 8), (1, 32, 4, 4)],
             'out': (1, 2720, 8), 'anchors': (2720, 4),
             'shapes': [('circle', 23.0, [3, 3, 3, 4]), ('square', 10.0, [2, 2, 2, 2]), ('triangle', 41.0, [4, 4, 4]),
                        ('circle', 20.0, [3, 3, 3]), ('circle', 10.5, [2, 2, 2])],
             'learn': 17, 'params': 299728, 'pyramid_params': 48384, 'head_params': 4624}

# What the course measured, on one thread, with the course's code (train_course): every detector of VARIANTS after
# torch.manual_seed(seed), the training scenes (spread=True) shuffled with a generator seeded with seed + 1, in
# mini-batches of BATCH, AdamW with a learning rate of LR, a warmup of one epoch and cosine decay, and gradients
# clipped at 1.0, with the loss of hard negative mining (hard_loss), for the seeds 0 to 4, as the one-stage detector
# of cv-one-stage-detector was trained. "map" is the mAP at an IoU of 0.5 over the 200 test scenes, with every box
# down to a score of LOW; "aps" the AP of each kind; "map75" the mAP at 0.75; "found" how many of the small, medium
# and large shapes of the test scenes get a box of their kind with a score of 0.5 or more and an IoU above 0.5;
# "false" how many boxes of 0.5 or more find no shape; "val" the loss on the test scenes after the last epoch. For
# the FPN, "levels" says for every level how many of the shapes it finds of each size come from that level's anchors.
# "curves" is the FPN with seed 0, the loss over every epoch, "scene" its boxes of 0.5 or more on the scene of the
# worked example (kind, score, box, anchor), "shapes" the small, medium and large shapes of the test scenes. "plain"
# is the FPN with seed 0 trained with the loss of the one-stage detector. tests/test_cv_fpn.py measures the FPN with
# seed 0 again; measure_trained() measures all of it.
TRAINED = {'fpn': {'map': [0.9411, 0.9338, 0.95, 0.9443, 0.9437],
         'aps': [[0.9497, 0.9363, 0.9372],
                 [0.9433, 0.9271, 0.931],
                 [0.9696, 0.9496, 0.9307],
                 [0.9525, 0.9426, 0.9379],
                 [0.9564, 0.9387, 0.936]],
         'map75': [0.8091, 0.7994, 0.8282, 0.8055, 0.788],
         'found': [[248, 256, 223], [244, 259, 225], [249, 269, 223], [245, 260, 227], [246, 265, 223]],
         'false': [110, 106, 107, 104, 114],
         'val': [0.2382, 0.2361, 0.2301, 0.2364, 0.2261],
         'levels': [{'2': [229, 8, 0], '3': [19, 233, 4], '4': [0, 15, 183], '5': [0, 0, 36]},
                    {'2': [224, 7, 0], '3': [20, 235, 5], '4': [0, 17, 182], '5': [0, 0, 38]},
                    {'2': [231, 3, 0], '3': [18, 240, 5], '4': [0, 26, 181], '5': [0, 0, 37]},
                    {'2': [226, 8, 0], '3': [19, 238, 5], '4': [0, 14, 185], '5': [0, 0, 37]},
                    {'2': [228, 11, 0], '3': [18, 238, 3], '4': [0, 16, 182], '5': [0, 0, 38]}]},
 'curves': {'train': [1.0233, 0.6236, 0.4114, 0.3018, 0.2217, 0.1602, 0.1078, 0.0844],
            'val': [0.7223, 0.7276, 0.5839, 0.2991, 0.305, 0.2467, 0.2398, 0.2382]},
 'scene': [(2, 1.0, (76.1, 63.3, 115.6, 103.0), 2651),
           (0, 1.0, (72.6, 11.4, 90.4, 29.7), 2373),
           (0, 1.0, (30.2, 41.6, 53.0, 63.8), 2221),
           (1, 0.995, (100.1, 106.6, 109.2, 115.9), 1721),
           (0, 0.991, (96.2, 31.3, 105.0, 39.3), 1617)],
 'shapes': [268, 282, 236],
 'bottom-up': {'map': [0.9478, 0.9492, 0.9465, 0.9537, 0.9568],
               'aps': [[0.962, 0.9477, 0.9336],
                       [0.9628, 0.9527, 0.9321],
                       [0.9659, 0.9385, 0.9352],
                       [0.9627, 0.9539, 0.9445],
                       [0.9767, 0.954, 0.9396]],
               'map75': [0.8005, 0.8135, 0.8035, 0.8173, 0.8204],
               'found': [[242, 266, 224], [252, 264, 221], [245, 263, 223], [251, 261, 227], [251, 265, 223]],
               'false': [99, 96, 113, 95, 95],
               'val': [0.2156, 0.2042, 0.2088, 0.2077, 0.2097]},
 'deep': {'map': [0.7608, 0.7905, 0.7783, 0.7624, 0.7657],
          'aps': [[0.8408, 0.7389, 0.7028],
                  [0.8581, 0.7941, 0.7193],
                  [0.8672, 0.7369, 0.7306],
                  [0.8389, 0.7593, 0.6891],
                  [0.8582, 0.7442, 0.6947]],
          'map75': [0.4714, 0.4707, 0.4306, 0.4734, 0.4484],
          'found': [[129, 206, 218], [143, 207, 223], [129, 208, 225], [130, 201, 223], [134, 215, 221]],
          'false': [96, 90, 106, 101, 108],
          'val': [0.3137, 0.3081, 0.3135, 0.3078, 0.3161]},
 'sharp': {'map': [0.8277, 0.8444, 0.8142, 0.8295, 0.8223],
           'aps': [[0.8657, 0.815, 0.8024],
                   [0.887, 0.826, 0.8202],
                   [0.8647, 0.7732, 0.8047],
                   [0.8845, 0.7912, 0.813],
                   [0.8589, 0.8021, 0.8059]],
           'map75': [0.5877, 0.6082, 0.6007, 0.5822, 0.6371],
           'found': [[211, 253, 223], [219, 262, 225], [206, 257, 225], [216, 256, 224], [202, 262, 225]],
           'false': [505, 515, 546, 527, 531],
           'val': [0.4492, 0.4212, 0.4402, 0.442, 0.4189]},
 'plain': {'map': 0.3334, 'map75': 0.2768, 'found': [6, 2, 1], 'false': 2}}


def sizes(stride: int) -> tuple:
    """The anchors of every square of a level whose squares are `stride` pixels: 2 and 2√2 times as wide."""
    return ((2 * stride, 2 * stride), (2 * math.sqrt(2) * stride, 2 * math.sqrt(2) * stride))


ALL_SIZES = tuple(size for _, _, stride in LEVELS for size in sizes(stride))   # 8 sizes, from 8 to 91 pixels


@functools.lru_cache(maxsize=4)
def anchors(variant: str = "fpn") -> torch.Tensor:
    """The anchors of a detector, in the order of its rows: for the pyramids, level 2 to 5, each with its own 2 sizes,
    2,720 anchors; for one map, all 8 sizes on every square: of 16 pixels ("deep", C4) or of 4 ("sharp", C2)."""
    if variant in ("fpn", "bottom-up"):
        a = np.concatenate([scenes.anchor_boxes(cells, stride, sizes(stride)) for _, cells, stride in LEVELS])
    elif variant == "deep":
        a = scenes.anchor_boxes(8, 16, ALL_SIZES)
    elif variant == "sharp":
        a = scenes.anchor_boxes(32, 4, ALL_SIZES)
    else:
        raise ValueError(variant)
    return torch.tensor(a, dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def level_of() -> torch.Tensor:
    """For every anchor of the pyramid, its level: 2 for the first 2,048, then 3, 4 and 5."""
    return torch.cat([torch.full((cells * cells * 2,), k) for k, cells, _ in LEVELS])


@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    return scenes.scene(SEED, spread=True)


@functools.lru_cache(maxsize=2)
def scene_set(which: str) -> tuple:
    """The training or the test scenes, with small and large shapes, and their pictures as one tensor of bytes, shape
    (N, 3, 128, 128)."""
    ss = tuple(scenes.scene(seed, spread=True) for seed in (TRAIN if which == "train" else TEST))
    return ss, torch.tensor(np.stack([s.image for s in ss])).permute(0, 3, 1, 2).contiguous()


def all_targets(ss, a) -> tuple[torch.Tensor, torch.Tensor]:
    """The classes (N, A) and the offsets (N, A, 4) of scenes for the anchors a, as the targets of the one-stage
    detector."""
    classes, ts = [], []
    for s in ss:
        c, t = targets(a.clone(), *boxes_and_labels(s))
        classes.append(c)
        ts.append(t)
    return torch.stack(classes), torch.stack(ts)


@functools.lru_cache(maxsize=2)
def course_targets(variant: str = "fpn") -> tuple:
    a = anchors(variant)
    return all_targets(scene_set("train")[0], a), all_targets(scene_set("test")[0], a)


def size_of(box) -> float:
    """The side of a square with the area of the box: how large a shape is."""
    return math.sqrt((box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def third(box) -> int:
    """0 for a small shape, 1 for a medium one, 2 for a large one."""
    g = size_of(box)
    return 0 if g < SMALL else (1 if g < LARGE else 2)


# --- the course's versions --------------------------------------------------------------------------------------------

def _part(c_in: int, c_out: int, pool: bool = True) -> list:
    return ([nn.MaxPool2d(2)] if pool else []) + [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]


class Backbone(nn.Module):
    """The backbone of the one-stage detector with a fourth pooling, in four parts: C2, 32 maps of 32 by 32 (every
    square 4 pixels); C3, 64 of 16 by 16 (8 pixels); C4, 128 of 8 by 8 (16 pixels); C5, 128 of 4 by 4 (32 pixels)."""

    def __init__(self):
        super().__init__()
        self.c2 = nn.Sequential(*_part(3, 8, False), *_part(8, 16), *_part(16, 32))
        self.c3 = nn.Sequential(*_part(32, 64))
        self.c4 = nn.Sequential(*_part(64, 128))
        self.c5 = nn.Sequential(*_part(128, 128))

    def forward(self, x):
        c2 = self.c2(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        return [c2, c3, c4, self.c5(c4)]


def pyramid(cs, lateral, smooth):
    """The maps of the feature pyramid, as the learner's: from the top down, the 1×1 convolution of each map of the
    backbone plus the level above made twice as large (nearest neighbor); then the 3×3 convolution of every level.
    [P2, P3, P4, P5]."""
    p = lateral[3](cs[3])
    merged = [p]
    for i in (2, 1, 0):
        p = lateral[i](cs[i]) + F.interpolate(p, scale_factor=2, mode="nearest")
        merged.insert(0, p)
    return [smooth[i](merged[i]) for i in range(4)]


def bottom_up(cs, lateral, smooth):
    """A pyramid without the way down, the paper's "bottom-up pyramid", which imitates SSD: every map of the backbone on
    its own, a 1×1 and a 3×3 convolution."""
    return [smooth[i](lateral[i](cs[i])) for i in range(4)]


def head_rows(head, maps):
    """The rows of every anchor, as the learner's: one head on every level, each level's numbers as rows of 8, square
    by square, and the levels after each other: (N, A, 8)."""
    return torch.cat([head(p).permute(0, 2, 3, 1).reshape(len(p), -1, 8) for p in maps], dim=1)


class FPNDetector(nn.Module):
    """The course's detector with a feature pyramid: the backbone, a 1×1 convolution to D channels for each of its four
    maps, a 3×3 convolution for each level, and one head, a 3×3 convolution that gives each of the 2 anchors of a
    square 4 logits and 4 offsets, on every level. `pyramid_fn` and `rows_fn` are the course's or the learner's."""

    def __init__(self, pyramid_fn=pyramid, rows_fn=head_rows):
        super().__init__()
        self.backbone = Backbone()
        self.lateral = nn.ModuleList([nn.Conv2d(c, D, 1) for c in CHANNELS])
        self.smooth = nn.ModuleList([nn.Conv2d(D, D, 3, padding=1) for _ in CHANNELS])
        self.head = nn.Conv2d(D, 2 * 8, 3, padding=1)
        self.pyramid_fn, self.rows_fn = pyramid_fn, rows_fn

    def forward(self, x):
        return self.rows_fn(self.head, self.pyramid_fn(self.backbone(x), self.lateral, self.smooth))


class OneMap(nn.Module):
    """For the comparison: the same backbone, and all 8 anchor sizes on one of its maps, C2 ("sharp", squares of 4
    pixels) or C4 ("deep", squares of 16), with a 1×1 and a 3×3 convolution as one level of the pyramid."""

    def __init__(self, level: int):
        super().__init__()
        self.level = level - 2
        self.backbone = Backbone()
        self.lateral = nn.Conv2d(CHANNELS[self.level], D, 1)
        self.smooth = nn.Conv2d(D, D, 3, padding=1)
        self.head = nn.Conv2d(D, len(ALL_SIZES) * 8, 3, padding=1)

    def forward(self, x):
        return head_rows(self.head, [self.smooth(self.lateral(self.backbone(x)[self.level]))])


def make(variant: str):
    if variant == "fpn":
        return FPNDetector()
    if variant == "bottom-up":
        return FPNDetector(pyramid_fn=bottom_up)
    return OneMap({"deep": 4, "sharp": 2}[variant])


def hard_loss(out, classes, t, ratio: float = HARD):
    """The loss of the detector with hard negative mining, as SSD: the cross-entropy over every anchor that learns a
    shape and, in each scene, the `ratio` times as many background anchors with the highest cross-entropy; plus the
    mean absolute difference of the offsets over the anchors that learn a shape."""
    bg, obj = classes == 0, classes > 0
    ce = F.cross_entropy(out[:, :, :4].reshape(-1, 4), classes.clamp(min=0).reshape(-1), reduction="none").reshape(classes.shape)
    rank = ce.detach().masked_fill(~bg, -1.0).argsort(dim=1, descending=True).argsort(dim=1)
    keep = obj | (bg & (rank < ratio * obj.sum(dim=1, keepdim=True)))
    return ce[keep].mean() + (out[:, :, 4:][obj] - t[obj]).abs().mean()


def train(model, variant: str = "fpn", seed: int = 0, heard=None, loss_fn=hard_loss) -> list[dict]:
    """Train `model` on the training scenes as the one-stage detector trained, with the loss of hard negative mining,
    and measure the validation loss on the test scenes after every epoch."""
    (C, T), (Ct, Tt) = course_targets(variant)
    _, X = scene_set("train")
    _, Xt = scene_set("test")
    return _ONE.train(model, loss_fn, (X, C, T), (Xt, Ct, Tt), seed, EPOCHS, heard)


def train_course(variant: str = "fpn", seed: int = 0, heard=None, loss_fn=hard_loss) -> tuple:
    """A detector of the course, trained with `seed` on one thread, as a check runs: (model, history)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = make(variant)
        history = train(model, variant, seed, heard, loss_fn)
    finally:
        torch.set_num_threads(threads)
    return model, history


# --- detecting and counting -------------------------------------------------------------------------------------------

def detect(out, a, score: float = LOW, threshold: float = NMS_IOU, boxes_fn=boxes_from) -> list[tuple]:
    """The boxes of one scene, (kind, score, box, anchor), the highest score first: every anchor with a score of `score`
    or more, turned into its box with boxes_fn, then NMS for each kind on its own."""
    s, k = scored(out)
    keep = (s >= score).nonzero().flatten()
    boxes = torch.as_tensor(boxes_fn(a[keep].clone(), out[keep, 4:].clone()), dtype=torch.float32).reshape(-1, 4)
    found = []
    for kind in range(KINDS):
        mine = (k[keep] == kind).nonzero().flatten()
        for j in scenes.nms(boxes[mine], s[keep][mine], threshold):
            i = int(mine[j])
            found.append((kind, float(s[keep][i]), tuple(boxes[i].tolist()), int(keep[i])))
    found.sort(key=lambda d: -d[1])
    return found


def shown(found: list) -> list[tuple]:
    """The boxes with a score of SCORE or more, rounded as the lesson writes them: (kind, score, box, anchor)."""
    return [(k, round(s, 3), tuple(round(v, 1) for v in b), i) for k, s, b, i in found if s >= SCORE]


def summary(out, ss, a, boxes_fn=boxes_from) -> dict:
    """What the lesson says of a trained detector, from its rows on the test scenes: the mAP at 0.5 and its AP per kind,
    the mAP at 0.75, the shapes it finds of each size (a box of their kind, a score of 0.5 or more, an IoU above 0.5),
    the shapes of each size, its false boxes of 0.5 or more, and for the pyramid, how many of the shapes it finds of
    each size each level finds."""
    found = [detect(out[i], a, boxes_fn=boxes_fn) for i in range(len(ss))]
    m, aps = mean_ap([[d[:3] for d in f] for f in found], ss)
    m75, _ = mean_ap([[d[:3] for d in f] for f in found], ss, 0.75)
    got, total, false = [0, 0, 0], [0, 0, 0], 0
    by_level = {k: [0, 0, 0] for k, _, _ in LEVELS} if len(a) == len(level_of()) else None
    for f, s in zip(found, ss):
        top = [d for d in f if d[1] >= SCORE]
        taken = set()
        for (kind, _, box, i), hit in zip(top, hits([d[:3] for d in top], s)):
            if not hit:
                false += 1
                continue
            q = [q for q, sh in enumerate(s.shapes) if sh.label == kind and iou(box, sh.box) > AP_IOU][-1]   # as hits()
            taken.add(q)
            if by_level is not None:
                by_level[int(level_of()[i])][third(s.shapes[q].box)] += 1
        for q, sh in enumerate(s.shapes):
            total[third(sh.box)] += 1
            got[third(sh.box)] += q in taken
    out = {"map": round(m, 4), "aps": [round(v, 4) for v in aps], "map75": round(m75, 4), "found": got, "shapes": total,
           "false": false}
    return out | ({"levels": by_level} if by_level is not None else {})


def measure_variant(variant: str, seed: int, loss_fn=hard_loss) -> dict:
    """Train one detector and measure what TRAINED holds for it; for the FPN with seed 0 also its curves, and its boxes
    on the scene of the worked example."""
    model, history = train_course(variant, seed, loss_fn=loss_fn)
    ss, Xt = scene_set("test")
    o = outputs(model, Xt)
    a = anchors(variant)
    out = summary(o, ss, a) | {"val": round(history[-1]["val_loss"], 4)}
    if variant == "fpn" and seed == 0 and loss_fn is hard_loss:
        out["curves"] = {"train": [round(h["train_loss"], 4) for h in history], "val": [round(h["val_loss"], 4) for h in history]}
        out["scene"] = shown(detect(o[TEST.index(SEED)], a))
    return out


# --- the worked example ---------------------------------------------------------------------------------------------

def worked() -> dict:
    """The worked example's numbers, as its cell computes them, with an untrained network after torch.manual_seed(0):
    the shapes of the maps, nearest-neighbor upsampling of a small map, the rows and anchors, and for every shape of
    the scene its kind, its size and the levels of the anchors that learn it."""
    s = example()
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    torch.manual_seed(0)
    net = FPNDetector()
    with torch.no_grad():
        cs = net.backbone(x)
        small = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        up = F.interpolate(small, scale_factor=2, mode="nearest")
        maps = pyramid(cs, net.lateral, net.smooth)
        out = head_rows(net.head, maps)
    a = anchors()
    label = torch.as_tensor(scenes.match_anchors(a.numpy(), [sh.box for sh in s.shapes]))
    shapes = []
    for k, sh in enumerate(s.shapes):
        mine = level_of()[label == k]
        shapes.append((sh.kind, round(size_of(sh.box), 1), mine.tolist()))
    return {"c": [tuple(c.shape) for c in cs], "up": up[0, 0].tolist(), "p": [tuple(p.shape) for p in maps],
            "out": tuple(out.shape), "anchors": tuple(a.shape), "shapes": shapes,
            "learn": int((label >= 0).sum()), "params": sum(v.numel() for v in net.parameters()),
            "pyramid_params": sum(v.numel() for v in [*net.lateral.parameters(), *net.smooth.parameters()]),
            "head_params": sum(v.numel() for v in net.head.parameters())}


def measure() -> dict:
    return worked()


# --- pictures ---------------------------------------------------------------------------------------------------------

def _escape(text: str) -> str:
    import html
    return html.escape(text, quote=True)


def pyramid_picture(words: dict) -> str:
    """The feature pyramid network, drawn: on the left the maps C2 to C5 of the backbone, each half as large as the one
    below; a 1×1 convolution of each to the right; from the top down, each level made twice as large and added to the
    level below; a 3×3 convolution to P2 to P5; and one head for all levels."""
    sides = {5: 14, 4: 22, 3: 34, 2: 50}
    tops = {5: 40, 4: 76, 3: 120, 2: 176}
    lx, nx, rx = 50, 140, 206   # the middles of the backbone's maps, of the nodes of the way down, of the pyramid's maps
    ink, ink2, ink3 = "var(--ea-ink)", "var(--ea-ink-2)", "var(--ea-ink-3)"
    fill = "fill:var(--ea-hidden-wash);stroke:var(--ea-hidden);stroke-width:1.5"
    text = "font-size:14px;fill:" + ink
    small = "font-size:13px;fill:" + ink2
    parts = []

    def arrow(x0, y0, x1, y1):
        """A line from (x0, y0) to (x1, y1) with a head at its end."""
        d = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        bx, by = x1 - 6 * ux, y1 - 6 * uy
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" style="stroke:{ink3};stroke-width:1.5"/>'
                     f'<path d="M{bx - 4 * uy:.1f} {by + 4 * ux:.1f} L{x1:.1f} {y1:.1f} L{bx + 4 * uy:.1f} {by - 4 * ux:.1f} Z" style="fill:{ink3}"/>')

    parts.append(f'<text x="{lx}" y="16" style="{text};font-weight:600;text-anchor:middle">{_escape(words["up"])}</text>'
                 f'<text x="{(nx + rx) / 2 + 12}" y="16" style="{text};font-weight:600;text-anchor:middle">{_escape(words["down"])}</text>')
    for k in (5, 4, 3, 2):
        side, top = sides[k], tops[k]
        mid = top + side / 2
        cells = {5: 4, 4: 8, 3: 16, 2: 32}[k]
        parts.append(f'<rect x="{lx - side / 2}" y="{top}" width="{side}" height="{side}" rx="2" style="{fill}"/>'
                     f'<text x="{lx - side / 2 - 6}" y="{mid + 5}" style="{text};text-anchor:end">C{k}</text>'
                     f'<rect x="{rx - side / 2}" y="{top}" width="{side}" height="{side}" rx="2" style="{fill}"/>'
                     f'<text x="{rx + side / 2 + 6}" y="{mid - 1}" style="{text}">P{k}</text>'
                     f'<text x="{rx + side / 2 + 6}" y="{mid + 14}" style="{small}">{cells} × {cells}</text>')
        arrow(lx + side / 2 + 3, mid, nx - (8 if k < 5 else 3), mid)                      # the 1×1 convolution
        parts.append(f'<text x="{lx + side / 2 + 16}" y="{mid - 6}" style="{small};text-anchor:middle">1×1</text>')
        if k < 5:
            parts.append(f'<circle cx="{nx}" cy="{mid}" r="7" style="fill:var(--ea-page);stroke:{ink2};stroke-width:1.5"/>'
                         f'<path d="M{nx - 4} {mid} H{nx + 4} M{nx} {mid - 4} V{mid + 4}" style="stroke:{ink2};stroke-width:1.5"/>')
            above = tops[k + 1] + sides[k + 1] / 2
            arrow(nx, above + (7 if k + 1 < 5 else 3), nx, mid - 8)                      # twice as large, down
            parts.append(f'<text x="{nx - 7}" y="{(above + mid) / 2 + 5}" style="{small};text-anchor:end">×2</text>')
        else:
            parts.append(f'<circle cx="{nx}" cy="{mid}" r="3" style="fill:{ink2}"/>')
        arrow(nx + (8 if k < 5 else 4), mid, rx - side / 2 - 3, mid)                     # the 3×3 convolution
        parts.append(f'<text x="{(nx + rx - side / 2) / 2 + 2}" y="{mid - 6}" style="{small};text-anchor:middle">3×3</text>')
        if k > 2:   # the backbone goes up: pooling and a convolution
            arrow(lx, tops[k - 1] - 3, lx, top + side + 3)
    # one head for every level: a bracket to the right of the pyramid's labels, and its words under it
    bx = 300
    y0, y1 = tops[5] + sides[5] / 2, tops[2] + sides[2] / 2
    parts.append(f'<path d="M{bx - 5} {y0} H{bx} V{y1} H{bx - 5}" style="fill:none;stroke:{ink3};stroke-width:1.5"/>')
    bottom = tops[2] + sides[2]
    for i, line in enumerate(words["head"]):
        parts.append(f'<text x="{bx + 4}" y="{bottom + 22 + 17 * i}" style="{text};text-anchor:end">{_escape(line)}</text>')
    width, height = 310, bottom + 22 + 17 * (len(words["head"]) - 1) + 8
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{width}px)">'
            f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{_escape(words["alt"])}</title>{"".join(parts)}</svg></figure>')


def _scene_svg(s, boxes, title, names=None, figsize=(3.9, 4.05), crop=(0, 0, 127, 127), size=10, labels=True):
    """One scene with its true boxes and the detector's boxes (kind, score, box, ...), as the one-stage detector draws
    them; with `names`, each box with its kind and score, and words that would stick out on the right end at the box's
    right side instead, since the shapes of these scenes lie close to the edge too."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    fig, ax = plt.subplots(figsize=figsize)
    _ONE._photo_axes(ax, s.image, title, crop=crop, size=size)
    for shape in s.shapes:
        _ONE._truth_box(ax, shape.box)
    for b in boxes:
        _ONE._found_box(ax, b[2])
    if names is not None and labels:
        placed, renderer = [], fig.canvas.get_renderer()
        room = ax.get_window_extent(renderer)
        for k, sc, b, *_ in boxes:
            words = _ONE._score(ax, b, names["box"].format(kind=names["kinds"][k], score=f"{sc:.2f}".replace(".", names["point"])))
            if words.get_window_extent(renderer).x1 > room.x1:
                words.set_x(b[3] + 0.5)
                words.set_ha("right")
            _ONE._clear_of(ax, words, b, placed, renderer)
            placed.append(words)
    svg = plots.render(fig)
    plt.close(fig)
    return svg


def level_boxes(k: int) -> list[tuple]:
    """The boxes of 0.5 or more of the course's trained detector (seed 0) on the scene of the worked example whose
    anchor lies on level k."""
    return [b for b in TRAINED["scene"] if int(level_of()[b[3]]) == k]


def level_frame(k: int, words: dict) -> tuple[str, dict]:
    """For the slider: the boxes on the scene of the worked example that come from anchors of level k, and next to it,
    how many shapes of each size that level finds on the 200 test scenes, of all the shapes the detector finds."""
    mine = level_boxes(k)
    title = words["title"][0 if len(mine) == 1 else 1]   # "1 box", "2 boxes"
    scene = _scene_svg(example(), mine, title.format(k=k, n=len(mine)), names=words, figsize=(2.9, 3.0), size=9)
    counts = TRAINED["fpn"]["levels"][0][str(k)]
    found = TRAINED["fpn"]["found"][0]
    xs = [0, 1, 2]
    top = max(found) * 1.15
    p = chart.Panel((-0.6, 2.6), (0, top), title=words["chart"].format(k=k), ylabel=words["shapes"], xticks=xs,
                    xtick_labels=list(words["thirds"]))
    for x, n, total in zip(xs, counts, found):
        p.rect(x - 0.32, 0, x + 0.32, total, "ink-3")    # every shape the detector finds of this size
        p.rect(x - 0.32, 0, x + 0.32, n, "ink")          # those this level finds
        inside = n > top * 0.2                           # its number inside the bar, under its top, or just above it
        p.text(x, n - top * 0.1 if inside else n + top * 0.03, str(n), "ink")
    plot = chart.figure([p], alt=words["chart_alt"].format(k=k), width=276, height=236)   # 11 px text in a play box on a phone
    if words["point"] == ",":
        plot = chart.with_decimal_comma(plot)
    stop = {"boxes": len(mine), "found": counts, "anchors": {2: 2048, 3: 512, 4: 128, 5: 32}[k],
            "sizes": [round(sz[0], 1) for sz in sizes(2 ** k)]}
    return _ONE._side_by_side([(scene, 250), (plot, 276)], words["alt"].format(k=k, n=len(mine))), stop


def comparison_table(words: dict, lang: str) -> str:
    """The four detectors of the course side by side, as a Markdown table: for each, its anchors, the small, medium and
    large shapes of the test scenes it finds, its mAP at 0.5 and 0.75, and its false boxes, from the lowest to the
    highest over the seeds 0 to 4."""
    def span(values, show):
        low, high = min(values), max(values)
        return show(low) if low == high else f"{show(low)}–{show(high)}"

    rows = ["| " + " | ".join(words["header"]) + " |", "|" + "---|" * len(words["header"])]
    for variant in VARIANTS:
        r = TRAINED[variant]
        cells = [words["names"][variant], whole(len(anchors(variant)), lang)]
        cells += [span([f[i] for f in r["found"]], str) for i in range(3)]
        cells += [span(r["map"], lambda x: number(x, 3, lang)), span(r["map75"], lambda x: number(x, 3, lang)),
                  span(r["false"], str)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    return _ONE.curves_picture(curves, labels, lang, at)


def rewind_picture(boxes: list, words: dict) -> str:
    """The test scenes of the rewind, each with its true boxes and the boxes of 0.5 or more that the learner's detector
    gave it at one moment, after NMS. words["scene"] is one label, or two: for one box, and for any other number."""
    panels = []
    for seed, found in zip(SHOWN, boxes):
        s = scenes.scene(seed, spread=True)
        label = words["scene"] if isinstance(words["scene"], str) else words["scene"][0 if len(found) == 1 else 1]
        panels.append((_scene_svg(s, found, label.format(seed=seed, n=len(found)), figsize=(2.3, 2.45),
                                  size=9, labels=False), 160))
    return _ONE._side_by_side(panels, words["alt"])


# --- exercise 2: pyramid ----------------------------------------------------------------------------------------------

_DOTS = _ONE._DOTS
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close

_Y_HEAD = ("import torch.nn.functional as F\n\n\ndef pyramid(cs, lateral, smooth):\n"
           "    # cs: the maps of the backbone, finest first, each half as large as the one before\n"
           "    # lateral, smooth: one 1x1 and one 3x3 convolution per level, finest first\n"
           "    # give back the levels of the pyramid, finest first: from the top down, the 1x1 map of\n"
           "    # each level plus the level above made twice as large; then the 3x3 convolution of each\n")
_Y_START = _Y_HEAD + "    ...\n"
_Y_TOP = "    down = [lateral[-1](cs[-1])]   # the levels from the top down\n"
_Y_LOOP = "    for i in reversed(range(len(cs) - 1)):\n"
_Y_ADD = "        down.append(lateral[i](cs[i]) + F.interpolate(down[-1], scale_factor=2, mode=\"nearest\"))\n"
_Y_RETURN = "    return [smooth[i](down[-1 - i]) for i in range(len(cs))]\n"


def _y(top=_Y_TOP, loop=_Y_LOOP, add=_Y_ADD, ret=_Y_RETURN) -> str:
    return _Y_HEAD + top + loop + add + ret


_Y = _y()
_Y_MAPS = ("import torch\nfrom torch import nn\n\ntorch.manual_seed({seed})\n"
           "cs = [torch.randn({n}, c, s, s) for c, s in {levels}]\n"
           "lateral = [nn.Conv2d(c, {d}, 1) for c, s in {levels}]\n"
           "smooth = [nn.Conv2d({d}, {d}, 3, padding=1) for c, s in {levels}]\n"
           "with torch.no_grad():\n    ps = pyramid(cs, lateral, smooth)\n")
_Y_SCENARIOS = [
    # four levels, as the lesson's pyramid: the shapes, and the first numbers of the finest level
    _Y_MAPS.format(seed=0, n=1, levels="[(4, 16), (6, 8), (8, 4), (8, 2)]", d=3)
    + "[list(p.shape) for p in ps] + [ps[0][0, 0, 0].tolist(), ps[3][0, 2].tolist()]",
    # three levels, two scenes
    _Y_MAPS.format(seed=1, n=2, levels="[(5, 8), (7, 4), (9, 2)]", d=2)
    + "[list(p.shape) for p in ps] + [ps[0][1, 1, 3].tolist(), ps[1][0, 0].tolist()]",
    # the maps of the backbone of the lesson: 128 by 128 pixels, 32 channels per level
    _Y_MAPS.format(seed=2, n=1, levels="[(32, 32), (64, 16), (128, 8), (128, 4)]", d=32)
    + "[ps[0][0, 5, 10, :4].tolist(), ps[1][0, 31, 0, :4].tolist(), ps[2][0, 0, 7].tolist()]",
]
_Y_WRONG = {
    "bilinear": _y(add=_Y_ADD.replace('mode="nearest"', 'mode="bilinear"')),
    "no-smooth": _y(ret="    return [down[-1 - i] for i in range(len(cs))]\n"),
    "no-down": _y(add="        down.append(lateral[i](cs[i]))\n"),
    "smoothed-down": _y(top="    down = [smooth[-1](lateral[-1](cs[-1]))]\n",
                        add="        down.append(smooth[i](lateral[i](cs[i]) + F.interpolate(down[-1], scale_factor=2, mode=\"nearest\")))\n",
                        ret="    return [down[-1 - i] for i in range(len(cs))]\n"),
    "top-first": _y(ret="    return [smooth[len(cs) - 1 - i](down[i]) for i in range(len(cs))]\n"),
}


def _sizes_differ(e: BaseException) -> bool:
    """Two maps of different sizes added: they differ in rows or columns (dimension 2 or 3)."""
    return isinstance(e, RuntimeError) and re.search(r"must match the size of tensor b \(\d+\) at non-singleton dimension [23]", str(e)) is not None


def _channels_added(e: BaseException) -> bool:
    """Two maps of different channels added (dimension 1)."""
    return isinstance(e, RuntimeError) and re.search(r"must match the size of tensor b \(\d+\) at non-singleton dimension 1\b", str(e)) is not None


def _channels_differ(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and re.search(r"expected input\[.*\] to have \d+ channels", str(e)) is not None


_Y_MISTAKES = [
    Mistake("not-twice", error=_sizes_differ,
            en="Python stops when your code adds two maps: they are not the same size. Is the level above made twice as large first, with `F.interpolate(..., scale_factor=2, mode=\"nearest\")`? And does your loop go from the level below the top down to level 0, for any number of levels, adding the last level in `down` to the 1×1 map of the level below it?",
            nl="Python stopt als je code twee maps optelt: ze zijn niet even groot. Maak je het niveau erboven eerst twee keer zo groot, met `F.interpolate(..., scale_factor=2, mode=\"nearest\")`? En gaat je loop van het niveau onder de top naar niveau 0, voor elk aantal niveaus, en telt hij het laatste niveau in `down` op bij de 1×1-map van het niveau eronder?"),
    Mistake("channels-added", error=_channels_added,
            en="Python stops when your code adds two maps: they have different numbers of channels. A map of the backbone, `cs[i]`, has its own channels; the levels of the pyramid have 32. First the 1×1 convolution, `lateral[i](cs[i])`, then add the level above.",
            nl="Python stopt als je code twee maps optelt: ze hebben verschillende aantallen channels. Een map van de backbone, `cs[i]`, heeft zijn eigen channels; de niveaus van de pyramid hebben er 32. Eerst de 1×1-convolutie, `lateral[i](cs[i])`, en tel daarna het niveau erboven erbij op."),
    Mistake("four-only", error=lambda e: isinstance(e, IndexError) and "out of range" in str(e),
            en="Python stops: your code, or the check reading your levels, reaches for a level that is not there. Does your loop go down to level 0, so that there is a level for every map of `cs`? And does it work for three levels too, with `len(cs)` instead of a fixed number: `for i in reversed(range(len(cs) - 1)):`?",
            nl="Python stopt: je code, of de check die je niveaus leest, vraagt om een niveau dat er niet is. Gaat je loop door tot en met niveau 0, zodat er een niveau is voor elke map van `cs`? En werkt hij ook voor drie niveaus, met `len(cs)` in plaats van een vast getal: `for i in reversed(range(len(cs) - 1)):`?"),
    # stacking instead of adding: a level then has more channels than its smooth[i] takes (after the mistakes above,
    # which are known by their own errors)
    Mistake("cat", syntax=r"torch\.cat\(",
            en="Python stops, and your code uses `torch.cat`. The levels are added, not stacked: `lateral[i](cs[i])` has the same 32 channels as the level above, so that the two can be added number by number, with `+`. Stacked along the channels, a level gets more channels than its `smooth[i]` takes.",
            nl="Python stopt, en je code gebruikt `torch.cat`. De niveaus worden opgeteld, niet gestapeld: `lateral[i](cs[i])` heeft dezelfde 32 channels als het niveau erboven, zodat die twee getal voor getal kunnen worden opgeteld, met `+`. Gestapeld langs de channels krijgt een niveau meer channels dan zijn `smooth[i]` aanneemt."),
    Mistake("channels", error=_channels_differ,
            en="Python stops in a convolution: it gets another number of channels than it was made for. `lateral[i]` takes the map of the backbone, `cs[i]`, with its own channels, and makes 32 of them; `smooth[i]` takes a level of the pyramid, after adding. Does each get its own map?",
            nl="Python stopt in een convolutie: die krijgt een ander aantal channels dan waarvoor hij gemaakt is. `lateral[i]` neemt de map van de backbone, `cs[i]`, met zijn eigen channels, en maakt er 32 van; `smooth[i]` neemt een niveau van de pyramid, na het optellen. Krijgt elk zijn eigen map?"),
    Mistake("bilinear", wrong=_Y_WRONG["bilinear"],
            en=_S_EN + ": you make the level above larger with `mode=\"bilinear\"`, which mixes neighboring squares. FPN copies every number into 2 by 2 squares, nearest neighbor: `mode=\"nearest\"`.",
            nl=_S_NL + ": je maakt het niveau erboven groter met `mode=\"bilinear\"`, dat naburige vakjes mengt. FPN kopieert elk getal naar 2 bij 2 vakjes, nearest neighbor: `mode=\"nearest\"`."),
    Mistake("no-smooth", wrong=_Y_WRONG["no-smooth"],
            en=_S_EN + ": the 3×3 convolution is missing. After adding, every level goes through its own `smooth[i]`, which also smooths out the blocks of 2 by 2 that making the level twice as large leaves.",
            nl=_S_NL + ": de 3×3-convolutie ontbreekt. Na het optellen gaat elk niveau door zijn eigen `smooth[i]`, die ook de blokken van 2 bij 2 gladstrijkt die het vergroten maakt."),
    Mistake("no-down", wrong=_Y_WRONG["no-down"],
            en=_S_EN + ": nothing comes down from the levels above: every level is only its own 1×1 map. That is a pyramid without the way down. Add the level above, made twice as large: `+ F.interpolate(down[-1], scale_factor=2, mode=\"nearest\")`.",
            nl=_S_NL + ": er komt niets naar beneden van de niveaus erboven: elk niveau is alleen zijn eigen 1×1-map. Dat is een pyramid zonder de weg naar beneden. Tel het niveau erboven erbij op, twee keer zo groot gemaakt: `+ F.interpolate(down[-1], scale_factor=2, mode=\"nearest\")`."),
    Mistake("smoothed-down", wrong=_Y_WRONG["smoothed-down"],
            en=_S_EN + ": you pass the level after its 3×3 convolution down, so the levels below get it smoothed again and again. FPN passes the sum, before the 3×3: keep the sums in `down`, and do the 3×3 of every level at the end.",
            nl=_S_NL + ": je geeft het niveau na zijn 3×3-convolutie door naar beneden, dus de niveaus eronder krijgen het steeds opnieuw gladgestreken. FPN geeft de som door, vóór de 3×3: houd de sommen in `down`, en doe de 3×3 van elk niveau aan het eind."),
    Mistake("top-first", wrong=_Y_WRONG["top-first"],
            en=_S_EN + ": your levels come back top first. Give them back finest first, as `cs` comes in: the anchors and the rows go from the finest level to the top.",
            nl=_S_NL + ": je niveaus komen met de top eerst terug. Geef ze met de fijnste eerst terug, zoals `cs` binnenkomt: de anchors en de rijen gaan van het fijnste niveau naar de top."),
]

# --- exercise 3: head_rows --------------------------------------------------------------------------------------------

_H_HEAD = ("import torch\n\n\ndef head_rows(head, maps):\n"
           "    # head: one convolution for every level; maps: the levels, finest first, each (N, C, H, W)\n"
           "    # give back (N, A, 8): on every level, the head's numbers of each square as rows of 8,\n"
           "    # square by square, row by row; the levels after each other, the finest first\n")
_H_START = _H_HEAD + "    ...\n"
_H_BODY = "    return torch.cat([head(p).permute(0, 2, 3, 1).reshape(len(p), -1, 8) for p in maps], dim=1)\n"
_H = _H_HEAD + _H_BODY
_H_MAPS = ("import torch\nfrom torch import nn\n\ntorch.manual_seed({seed})\n"
           "head = nn.Conv2d({c}, {k} * 8, 3, padding=1)\n"
           "maps = [torch.randn({n}, {c}, s, s) for s in {sides}]\n"
           "with torch.no_grad():\n    out = head_rows(head, maps)\n")
_H_SCENARIOS = [
    # two levels of 8 by 8 and 4 by 4 squares, 2 anchors per square: 160 rows
    _H_MAPS.format(seed=0, c=3, k=2, n=2, sides="(8, 4)") + "[list(out.shape), out[1, 129].tolist(), out[0, 3].tolist()]",
    # three levels, one anchor per square, one scene
    _H_MAPS.format(seed=1, c=4, k=1, n=1, sides="(4, 2, 1)") + "[list(out.shape), out[0, 16].tolist(), out[0, 20].tolist()]",
    # the four levels of the lesson: 2,720 rows
    _H_MAPS.format(seed=2, c=32, k=2, n=1, sides="(32, 16, 8, 4)") + "[list(out.shape), out[0, 2047].tolist(), out[0, 2719].tolist()]",
]
_H_WRONG = {
    "no-permute": _H_HEAD + "    return torch.cat([head(p).reshape(len(p), -1, 8) for p in maps], dim=1)\n",
    "top-first": _H_HEAD + "    return torch.cat([head(p).permute(0, 2, 3, 1).reshape(len(p), -1, 8) for p in reversed(maps)], dim=1)\n",
    "columns-first": _H_HEAD + "    return torch.cat([head(p).permute(0, 3, 2, 1).reshape(len(p), -1, 8) for p in maps], dim=1)\n",
}


_H_MISTAKES = [
    Mistake("view", error=lambda e: isinstance(e, RuntimeError) and "view size is not compatible" in str(e),
            en="`.view` stops: after `permute`, the numbers no longer lie in memory in the order of the new axes. Use `.reshape`.",
            nl="`.view` stopt: na `permute` liggen de getallen in het geheugen niet meer in de volgorde van de nieuwe axes. Gebruik `.reshape`."),
    # reshape(1, -1, 8), as in step 4 of the worked example: the check then reads a second scene that is not there
    Mistake("one-scene", syntax=r"reshape\(\s*1\s*,",
            en="Python stops: your rows have room for one scene, and the check gives two at once. `reshape(1, -1, 8)` was right for the one scene of the worked example; make rows for every scene with `reshape(len(p), -1, 8)`.",
            nl="Python stopt: je rijen hebben plaats voor één scène, en de check geeft er twee tegelijk. `reshape(1, -1, 8)` klopte voor de ene scène van het uitgewerkte voorbeeld; maak rijen voor elke scène met `reshape(len(p), -1, 8)`."),
    Mistake("stack", error=lambda e: isinstance(e, RuntimeError) and "stack expects each tensor to be equal size" in str(e),
            en="`torch.stack` stops: it lays tensors of the same shape on top of each other, and the levels have different numbers of rows. `torch.cat(..., dim=1)` puts them after each other.",
            nl="`torch.stack` stopt: die legt tensors van dezelfde shape op elkaar, en de niveaus hebben verschillende aantallen rijen. `torch.cat(..., dim=1)` zet ze achter elkaar."),
    Mistake("scenes-axis", error=lambda e: isinstance(e, RuntimeError) and "must match except in dimension 0" in str(e),
            en="Python stops in `torch.cat`: it puts the rows of the levels under each other, as more scenes, and the levels have different numbers of rows. Put them after each other within every scene: `torch.cat(..., dim=1)`.",
            nl="Python stopt in `torch.cat`: die zet de rijen van de niveaus onder elkaar, als meer scènes, en de niveaus hebben verschillende aantallen rijen. Zet ze binnen elke scène achter elkaar: `torch.cat(..., dim=1)`."),
    Mistake("no-permute", wrong=_H_WRONG["no-permute"],
            en=_S_EN + ": the shape is right, but the rows do not belong to the anchors. The head gives (N, 16, H, W): first `permute(0, 2, 3, 1)`, so that the numbers of a square lie together, then reshape, as in your `Detector`.",
            nl=_S_NL + ": de shape klopt, maar de rijen horen niet bij de anchors. De head geeft (N, 16, H, W): eerst `permute(0, 2, 3, 1)`, zodat de getallen van een vakje bij elkaar liggen, en dan reshape, zoals in je `Detector`."),
    Mistake("top-first", wrong=_H_WRONG["top-first"],
            en=_S_EN + ": your rows start with the top level. The anchors start with the finest level, so the rows must too: `for p in maps`.",
            nl=_S_NL + ": je rijen beginnen met het bovenste niveau. De anchors beginnen met het fijnste niveau, dus de rijen ook: `for p in maps`."),
    Mistake("columns-first", wrong=_H_WRONG["columns-first"],
            en=_S_EN + ": your rows go column by column. The anchors go row by row: `permute(0, 2, 3, 1)` keeps the rows, axis 2, before the columns, axis 3.",
            nl=_S_NL + ": je rijen gaan kolom voor kolom. De anchors gaan rij voor rij: `permute(0, 2, 3, 1)` houdt de rijen, axis 2, vóór de kolommen, axis 3."),
]

EXERCISES = {
    "pyramid": ScenarioBox(
        name="pyramid",
        picture=yours.image(call=0, result="[p[0, 0] for p in ps]"),
        start=_Y_START,
        solution=_Y,
        scenarios=_Y_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "pyramid"),
        first=[_DOTS],
        mistakes=_Y_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the top level, the 1×1 map of the last map of `cs`; then, from the level below it down to the finest, the 1×1 map of each plus the level above made twice as large; and the 3×3 convolution of every level, finest first.",
                      "nl": "Vervang de `...` door je code: het bovenste niveau, de 1×1-map van de laatste map van `cs`; dan, van het niveau eronder tot het fijnste, de 1×1-map van elk plus het niveau erboven twee keer zo groot gemaakt; en de 3×3-convolutie van elk niveau, de fijnste eerst."},
            "wrong": {"en": _S_EN + ". Is the top level the 1×1 map of the last map? Does every level below it add the level above, made twice as large with nearest neighbor, before its 3×3 convolution? And do the levels come back finest first?",
                      "nl": _S_NL + ". Is het bovenste niveau de 1×1-map van de laatste map? Telt elk niveau eronder het niveau erboven erbij op, twee keer zo groot gemaakt met nearest neighbor, vóór zijn 3×3-convolutie? En komen de niveaus met de fijnste eerst terug?"},
            "almost": {"en": "Right for the four levels. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of levels?",
                       "nl": "Goed voor de vier niveaus. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal niveaus?"},
            "correct": {"en": "Your `pyramid` is right: every level gets what the levels above it know, at its own size. It is now in your project, as `{saved}`.",
                        "nl": "Je `pyramid` klopt: elk niveau krijgt wat de niveaus erboven weten, op zijn eigen grootte. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `pyramid` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `pyramid` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "head-rows": ScenarioBox(
        name="head-rows",
        start=_H_START,
        solution=_H,
        scenarios=_H_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "head_rows"),
        first=[_DOTS],
        mistakes=_H_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: for every level, the numbers the head gives each square, as rows of 8, as your `Detector` made them; then the rows of the levels after each other within every scene, the finest first.",
                      "nl": "Vervang de `...` door je code: voor elk niveau de getallen die de head elk vakje geeft, als rijen van 8, zoals je `Detector` ze maakte; en dan de rijen van de niveaus binnen elke scène achter elkaar, de fijnste eerst."},
            "wrong": {"en": _S_EN + ". Do the numbers of each square become rows of 8, square by square, row by row, on every level? And do the levels come after each other in every scene, the finest first?",
                      "nl": _S_NL + ". Worden de getallen van elk vakje rijen van 8, vakje voor vakje, rij voor rij, op elk niveau? En komen de niveaus in elke scène achter elkaar, de fijnste eerst?"},
            "almost": {"en": "Right for the two levels. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of levels, any number of anchors per square, and any number of scenes?",
                       "nl": "Goed voor de twee niveaus. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal niveaus, elk aantal anchors per vakje en elk aantal scènes?"},
            "correct": {"en": "Your `head_rows` is right: one head for every level, and one row for every anchor, in the order of the anchors. It is now in your project, as `{saved}`.",
                        "nl": "Je `head_rows` klopt: één head voor elk niveau, en één rij voor elke anchor, in de volgorde van de anchors. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `head_rows` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `head_rows` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "head" and "background" are choices, "why-top-down" the open question.
REQUIRED = ["head", "pyramid", "head-rows", "background", "why-top-down"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("pyramid", _Y_START, "empty", None),
    ("pyramid", _Y, "correct", None),
    # the level above in a variable of its own
    ("pyramid", _y(top="    p = lateral[-1](cs[-1])\n    down = [p]\n",
                   add="        p = lateral[i](cs[i]) + F.interpolate(p, scale_factor=2, mode=\"nearest\")\n        down.append(p)\n"),
     "correct", None),
    # the levels turned around with reversed first
    ("pyramid", _y(ret="    ps = [p for p in reversed(down)]\n    return [smooth[i](ps[i]) for i in range(len(cs))]\n"), "correct", None),
    # the level above as large as the one below, by its size
    ("pyramid", _y(add="        down.append(lateral[i](cs[i]) + F.interpolate(down[-1], size=cs[i].shape[-2:], mode=\"nearest\"))\n"),
     "correct", None),
    # four levels written out, as in the worked example: right for four, and stops at three
    ("pyramid", _Y_HEAD + "    p5 = lateral[3](cs[3])\n"
     "    p4 = lateral[2](cs[2]) + F.interpolate(p5, scale_factor=2, mode=\"nearest\")\n"
     "    p3 = lateral[1](cs[1]) + F.interpolate(p4, scale_factor=2, mode=\"nearest\")\n"
     "    p2 = lateral[0](cs[0]) + F.interpolate(p3, scale_factor=2, mode=\"nearest\")\n"
     "    return [smooth[0](p2), smooth[1](p3), smooth[2](p4), smooth[3](p5)]\n", "error", "four-only"),
    ("pyramid", _y(add="        down.append(lateral[i](cs[i]) + down[-1])\n"), "error", "not-twice"),
    # a loop without reversed: from level 0 up, the last level in down is not the one above
    ("pyramid", _y(loop="    for i in range(len(cs) - 1):\n"), "error", "not-twice"),
    # a loop for four levels: right for four, and the sizes do not fit at three
    ("pyramid", _y(loop="    for i in (2, 1, 0):\n"), "error", "not-twice"),
    # a loop that stops before level 0: the return reaches for a level that is not there
    ("pyramid", _y(loop="    for i in reversed(range(1, len(cs) - 1)):\n"), "error", "four-only"),
    # the levels stacked along the channels instead of added
    ("pyramid", "import torch\n" + _y(add="        down.append(torch.cat([lateral[i](cs[i]), F.interpolate(down[-1], scale_factor=2, mode=\"nearest\")], dim=1))\n"),
     "error", "cat"),
    ("pyramid", _y(add="        down.append(lateral[i](cs[i] + F.interpolate(down[-1], scale_factor=2, mode=\"nearest\")))\n"),
     "error", "channels-added"),
    ("pyramid", _y(top="    down = [smooth[-1](cs[-1])]\n"), "error", "channels"),
    ("pyramid", _y(ret="    ...\n"), "wrong", "dots-left"),
] + [("pyramid", code, "wrong", key) for key, code in _Y_WRONG.items()] + [
    ("head-rows", _H_START, "empty", None),
    ("head-rows", _H, "correct", None),
    # the rows of every level in a loop
    ("head-rows", _H_HEAD + "    rows = []\n    for p in maps:\n        out = head(p)\n"
     "        rows.append(out.permute(0, 2, 3, 1).reshape(out.shape[0], -1, 8))\n    return torch.cat(rows, dim=1)\n", "correct", None),
    ("head-rows", _H_HEAD + "    return torch.cat([head(p).permute(0, 2, 3, 1).view(len(p), -1, 8) for p in maps], dim=1)\n", "error", "view"),
    # one scene, as in step 4 of the worked example
    ("head-rows", _H_HEAD + "    return torch.cat([head(p).permute(0, 2, 3, 1).reshape(1, -1, 8) for p in maps], dim=1)\n", "error", "one-scene"),
    ("head-rows", _H_HEAD + "    return torch.stack([head(p).permute(0, 2, 3, 1).reshape(len(p), -1, 8) for p in maps], dim=1)\n", "error", "stack"),
    ("head-rows", _H_HEAD + "    return torch.cat([head(p).permute(0, 2, 3, 1).reshape(len(p), -1, 8) for p in maps])\n", "error", "scenes-axis"),
    # 2 anchors per square as a fixed number: right for two anchors, wrong for one
    ("head-rows", _H_HEAD + "    return torch.cat([head(p).permute(0, 2, 3, 1).reshape(len(p), p.shape[2] * p.shape[3] * 2, 8) for p in maps], dim=1)\n",
     "error", None),
] + [("head-rows", code, "wrong", key) for key, code in _H_WRONG.items()]


# --- the demo: the course's FPN detector with the learner's pyramid and head_rows ---------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


class _Stopped(Exception):
    """An error in one of the learner's functions while the demo runs, with the key of the note that names it."""

    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


def _named(fn, key: str):
    """fn, with every error it raises turned into _Stopped(key), so that the demo names the module that stopped; a
    Timeout goes on as it is."""
    def run(*args):
        try:
            return fn(*args)
        except Timeout:
            raise
        except Exception as e:
            raise _Stopped(key) from e
    return run


def _demo(show=None):
    """For the demo: None when pyramid.py or head_rows.py is not in the project yet; "pyramid" when the learner's
    pyramid stops or does not give the four levels of 32 maps that training can follow back; "head_rows" when their
    head_rows stops or does not give 2,720 rows of 8; "offsets" when their boxes_from stops on the trained detector's
    offsets; "slow" when the training takes too long; else a dict with the loss after every epoch ("curves"), the boxes
    on the SHOWN scenes after every epoch ("moments"), the summary of the trained detector on the test scenes, its
    parameters, and the modules the course's versions stood in for. `show` hears the run while it trains."""
    if not (_saved("pyramid") and _saved("head_rows")):
        return None
    names = {}
    for module, name, key in (("pyramid", "pyramid", "pyramid"), ("head_rows", "head_rows", "head_rows"),
                              ("offsets", "boxes_from", "offsets")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    _, X = scene_set("train")
    torch.manual_seed(0)
    net = FPNDetector()
    want = [(1, D, cells, cells) for _, cells, _ in LEVELS]
    try:   # one scene through both: the shapes the training needs
        cs = net.backbone(X[:1].float() / 255)
        maps = names["pyramid"](cs, net.lateral, net.smooth)
        if not (len(maps) == 4 and all(torch.is_tensor(p) and tuple(p.shape) == w and p.requires_grad for p, w in zip(maps, want))):
            return "pyramid"
    except Exception:
        return "pyramid"
    try:
        rows = names["head_rows"](net.head, pyramid(cs, net.lateral, net.smooth))
        if not (torch.is_tensor(rows) and tuple(rows.shape) == (1, len(anchors()), 8) and rows.requires_grad):
            return "head_rows"
    except Exception:
        return "head_rows"
    pyr, rows_of, boxes_of = (_named(names["pyramid"], "pyramid"), _named(names["head_rows"], "head_rows"),
                              _named(names["boxes_from"], "offsets"))
    torch.manual_seed(0)   # the same start as the course's run
    model = FPNDetector(pyr, rows_of)
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
        history = with_time_limit(lambda: train(model, "fpn", 0, heard), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    except RuntimeError:   # the levels of the learner's pyramid did not fit the head after all
        return "pyramid"
    if broken:
        return broken[0]
    test_ss, Xt = scene_set("test")
    try:
        result = summary(outputs(model, Xt), test_ss, a, boxes_fn=boxes_of)
    except _Stopped as e:   # the learner's module that stopped; any other error is the course's own
        return e.key
    curves = {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]}
    return result | {"curves": curves, "moments": moments, "params": sum(v.numel() for v in model.parameters()), "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("pyramid", "head_rows", "offsets"):
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


# --- measuring everything again -------------------------------------------------------------------------------------

def measure_trained(seeds=range(5)) -> dict:
    """Everything TRAINED holds, measured again: every detector of VARIANTS with every seed, and the FPN with seed 0
    trained with the loss of the one-stage detector. Takes two to three minutes per training on one thread."""
    out = {}
    for variant in VARIANTS:
        runs = [measure_variant(variant, seed) for seed in seeds]
        out[variant] = {key: [r[key] for r in runs] for key in ("map", "aps", "map75", "found", "false", "val")}
        if variant == "fpn":
            out[variant]["levels"] = [{str(k): v for k, v in r["levels"].items()} for r in runs]
            out["curves"], out["scene"], out["shapes"] = runs[0]["curves"], runs[0]["scene"], runs[0]["shapes"]
    plain = measure_variant("fpn", 0, loss_fn=_ONE.detection_loss)
    out["plain"] = {key: plain[key] for key in ("map", "map75", "found", "false")}
    return out
