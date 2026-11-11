# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-rcnn: roi_pool, which cuts the feature maps of one box to k by k numbers per channel (RoI
pooling), and proposals, which turns the rows of a region proposal network into the boxes worth a second look. Both
are saved in the learner's project (work/projects/cnn/roi_pool.py and proposals.py) when right. The demo trains the
course's two-stage detector with them on the course's scenes: the backbone of cv-one-stage-detector, a region proposal
network, and a second stage on the pooled features of every proposal; then it finds the shapes of new scenes with the
learner's boxes_from and NMS, and measures the mAP. And the pictures: the proposals and the boxes of the course's
trained two-stage detector on the scene of the worked example, RoI pooling of one proposal, and for the slider, what
the number of proposals does. Everything is computed from the course's own scenes (scenes.py): no download."""
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
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_rcnn",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The scenes, their anchors, the targets of every anchor, the backbone, the mAP and the pictures of boxes on scenes
_ONE = _lesson_checks("cv-one-stage-detector")

# The learner's proposals imports their boxes_from of cv-anchors; where it is missing or broken, the course's version
# stands in, and the lesson says so.
BUILDS_ON = (("cnn", "offsets", "cv-anchors", "boxes-from"),)
projects.builds_on(*BUILDS_ON)

SEED, KINDS, CELLS, STRIDE = _ONE.SEED, _ONE.KINDS, _ONE.CELLS, _ONE.STRIDE
N_TRAIN, N_TEST, TRAIN, TEST = _ONE.N_TRAIN, _ONE.N_TEST, _ONE.TRAIN, _ONE.TEST
EPOCHS, LR, BATCH = _ONE.EPOCHS, _ONE.LR, _ONE.BATCH
SCORE, NMS_IOU, LOW, AP_IOU = _ONE.SCORE, _ONE.NMS_IOU, _ONE.LOW, _ONE.AP_IOU
SHOWN, SHOWN_SHAPES = _ONE.SHOWN, _ONE.SHOWN_SHAPES
anchors, example, scene_set, boxes_and_labels = _ONE.anchors, _ONE.example, _ONE.scene_set, _ONE.boxes_and_labels
iou, mean_ap, hits, lr_factor = _ONE.iou, _ONE.mean_ap, _ONE.hits, _ONE.lr_factor
offsets, boxes_from, iou_matrix = _ONE.offsets, _ONE.boxes_from, _ONE.iou_matrix
number, whole, low_high = _ONE.number, _ONE.whole, _ONE.low_high

K = 4                         # RoI pooling cuts every proposal to 4 by 4 bins: 64 x 4 x 4 = 1,024 numbers
PER_RPN = 2 + 4               # the RPN gives every anchor 2 logits (background, object) and 4 offsets
PER = 1 + KINDS + 4           # the second stage gives every proposal 4 logits and 4 offsets from the proposal
HIDDEN = 128                  # the linear layer of the second stage
BEST, RPN_NMS = 100, 0.7      # proposals: the 100 anchors with the highest probability of an object, NMS at 0.7
TRAIN_KEEP = TEST_KEEP = 32   # the proposals kept per scene, in training (the true boxes join them) and to detect
FG_IOU = 0.5                  # a proposal learns a kind from an IoU of 0.5 on with a true box, else background
KEEPS = (1, 2, 4, 8, 16, 32, 64)   # the proposals kept to detect, on the slider; 32 is the lesson's
TRAIN_LIMIT = 900             # seconds the demo gives the training: the note says fifteen minutes
POOLED = 2                    # the box the worked example pools: the square of its scene, s.shapes[2]

# The numbers of the worked example, measured without training. `measure()` measures them again; the test of the
# lesson checks that both are the same.
REFERENCE = {'maps': (64, 16, 16), 'rows': (768, 6), 'kept_after_nms': 95, 'proposals': 32, 'first': (60.0, 43.4, 88.8, 84.2),
             'score': (0.8329, 0.0932), 'box': (42, 39, 73, 72), 'squares': (5, 4, 9, 9), 'region': (64, 5, 6),
             'bins': ([(0, 2), (1, 3), (2, 4), (3, 5)], [(0, 2), (1, 3), (3, 5), (4, 6)]),
             'pooled': (1.2122, 1.2122, 1.1315, 1.1315, 1.2122, 1.2122, 1.812, 1.812, 0.8164, 1.1989, 5.2293, 3.8161, 0.8669, 2.3434,
                        5.2293, 3.8161),
             'pooled_shape': (32, 64, 4, 4), 'out': (32, 8), 'params': 167386, 'second_params': 132232, 'rpn_params': 10386}

# What the course measured, on one thread, with the course's code (train_course): TwoStage after
# torch.manual_seed(seed), the training scenes shuffled with a generator seeded with seed + 1, in mini-batches of
# BATCH, AdamW with a learning rate of LR, a warmup of one epoch and cosine decay, and gradients clipped at 1.0, for
# the seeds 0 to 4, as the one-stage detector of cv-one-stage-detector was trained. "map" is the mAP at an IoU of 0.5
# over the 200 test scenes, with every box down to a score of LOW; "aps" the AP of each kind; "map75" the mAP at 0.75;
# "found" how many of the shapes of the test scenes get a box of their kind with a score of 0.5 or more and an IoU
# above 0.5, "false" how many boxes of 0.5 or more find no shape; "covered" how many shapes have a proposal with an
# IoU above 0.5. "curves" is seed 0, the loss of each stage over every epoch. "scene" is seed 0 on the scene of the
# worked example: its proposals, the highest probability first, and the boxes of 0.5 or more after NMS. "keeps" is
# seed 0 with another number of proposals kept to detect. "align" is seed 0 trained and tested with roi_align instead
# of RoI pooling, at every number of proposals kept. tests/test_cv_rcnn.py measures seed 0 again; measure_trained()
# measures all of it.
TRAINED = {'map': [0.9593, 0.962, 0.9466, 0.9546, 0.9558],
 'aps': [[0.9666, 0.9637, 0.9476],
         [0.9825, 0.9622, 0.9414],
         [0.9706, 0.9561, 0.9129],
         [0.9648, 0.9643, 0.9345],
         [0.9666, 0.9689, 0.9319]],
 'map75': [0.7791, 0.7576, 0.7434, 0.7335, 0.7659],
 'found': [769, 773, 766, 761, 776],
 'false': [115, 107, 116, 116, 108],
 'covered': [805, 805, 804, 805, 804],
 'shapes': 808,
 'curves': {'rpn': [0.2392, 0.0998, 0.0859, 0.0744, 0.0665, 0.0591, 0.0538, 0.0505],
            'second': [1.1287, 0.6999, 0.5356, 0.4765, 0.4274, 0.3948, 0.365, 0.3496]},
 'scene': {'after': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)),
                     (2, 0.984, (52.1, 7.6, 74.3, 28.9)),
                     (1, 0.948, (43.8, 38.2, 71.7, 65.2)),
                     (2, 0.923, (99.1, 7.3, 117.4, 28.0))],
           'proposals': [(0.999, (52.3, 6.8, 75.2, 28.6)),
                         (0.998, (58.1, 49.4, 86.6, 78.9)),
                         (0.994, (41.6, 40.6, 66.3, 64.6)),
                         (0.967, (43.5, 38.4, 73.4, 67.4)),
                         (0.92, (39.5, 42.4, 69.0, 72.8)),
                         (0.743, (43.2, 53.8, 62.0, 74.6)),
                         (0.729, (44.0, 41.2, 75.5, 73.6)),
                         (0.729, (100.3, 6.4, 117.0, 22.6)),
                         (0.352, (41.4, 41.4, 61.4, 60.9)),
                         (0.266, (51.4, 40.4, 81.0, 69.1)),
                         (0.2, (102.4, 8.1, 120.6, 24.5)),
                         (0.189, (47.2, 49.9, 75.1, 78.1)),
                         (0.086, (48.7, 36.2, 72.2, 57.9)),
                         (0.078, (54.3, 35.9, 75.4, 55.9)),
                         (0.073, (56.6, 37.1, 73.6, 52.9)),
                         (0.047, (43.4, 37.8, 65.3, 59.5)),
                         (0.046, (51.8, 8.9, 76.0, 35.0)),
                         (0.027, (53.3, 40.9, 77.6, 63.2)),
                         (0.022, (105.6, 8.5, 117.7, 22.4)),
                         (0.021, (98.5, 6.5, 118.3, 29.3)),
                         (0.019, (65.1, 58.9, 86.5, 77.9)),
                         (0.018, (42.1, 53.6, 59.6, 70.9)),
                         (0.017, (55.8, 6.3, 73.2, 21.5)),
                         (0.016, (45.4, 39.6, 62.0, 53.9)),
                         (0.015, (50.5, 38.6, 70.7, 52.2)),
                         (0.014, (52.3, 12.4, 72.2, 33.2)),
                         (0.013, (51.4, 33.7, 77.8, 59.0)),
                         (0.011, (57.8, 43.6, 84.2, 69.8)),
                         (0.009, (64.9, 48.5, 85.0, 69.3)),
                         (0.008, (102.2, 3.7, 119.1, 22.3)),
                         (0.007, (93.1, -12.6, 120.7, 14.0)),
                         (0.007, (41.9, 41.1, 53.3, 61.3))]},
 'keeps': {1: {'aps': [0.3275, 0.3156, 0.0805],
               'boxes': [(2, 0.984, (52.1, 7.6, 74.3, 28.9))],
               'covered': 203,
               'false': 3,
               'found': 197,
               'map': 0.2412,
               'map75': 0.2335},
           2: {'aps': [0.6439, 0.5626, 0.2375],
               'boxes': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)), (2, 0.984, (52.1, 7.6, 74.3, 28.9))],
               'covered': 404,
               'false': 5,
               'found': 392,
               'map': 0.4813,
               'map75': 0.4588},
           4: {'aps': [0.9184, 0.8918, 0.7585],
               'boxes': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)),
                         (2, 0.984, (52.1, 7.6, 74.3, 28.9)),
                         (1, 0.948, (43.8, 38.2, 71.7, 65.2))],
               'covered': 717,
               'false': 13,
               'found': 693,
               'map': 0.8562,
               'map75': 0.7831},
           8: {'aps': [0.9427, 0.9547, 0.9208],
               'boxes': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)),
                         (2, 0.984, (52.1, 7.6, 74.3, 28.9)),
                         (1, 0.948, (43.8, 38.2, 71.7, 65.2)),
                         (2, 0.858, (100.3, 8.2, 117.3, 24.6))],
               'covered': 787,
               'false': 32,
               'found': 755,
               'map': 0.9394,
               'map75': 0.8235},
           16: {'aps': [0.9699, 0.9694, 0.9457],
                'boxes': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)),
                          (2, 0.984, (52.1, 7.6, 74.3, 28.9)),
                          (1, 0.948, (43.8, 38.2, 71.7, 65.2)),
                          (2, 0.858, (100.3, 8.2, 117.3, 24.6))],
                'covered': 801,
                'false': 57,
                'found': 767,
                'map': 0.9617,
                'map75': 0.8007},
           32: {'aps': [0.9666, 0.9637, 0.9476],
                'boxes': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)),
                          (2, 0.984, (52.1, 7.6, 74.3, 28.9)),
                          (1, 0.948, (43.8, 38.2, 71.7, 65.2)),
                          (2, 0.923, (99.1, 7.3, 117.4, 28.0))],
                'covered': 805,
                'false': 115,
                'found': 769,
                'map': 0.9593,
                'map75': 0.7791},
           64: {'aps': [0.9657, 0.9596, 0.9384],
                'boxes': [(0, 0.991, (57.5, 48.3, 87.4, 79.2)),
                          (2, 0.984, (52.1, 7.6, 74.3, 28.9)),
                          (1, 0.948, (43.8, 38.2, 71.7, 65.2)),
                          (2, 0.923, (99.1, 7.3, 117.4, 28.0))],
                'covered': 805,
                'false': 166,
                'found': 769,
                'map': 0.9546,
                'map75': 0.7512}},
 'align': {'map': 0.9639,
           'map75': 0.8912,
           'found': 764,
           'false': 47,
           'keeps': {1: {'map': 0.242, 'map75': 0.239},
                     2: {'map': 0.4812, 'map75': 0.4679},
                     4: {'map': 0.8528, 'map75': 0.8024},
                     8: {'map': 0.9436, 'map75': 0.8755},
                     16: {'map': 0.9564, 'map75': 0.8835},
                     32: {'map': 0.9639, 'map75': 0.8912},
                     64: {'map': 0.9667, 'map75': 0.8923}}}}


# --- the course's versions --------------------------------------------------------------------------------------------

def roi_pool(maps, box, k: int = K):
    """RoI pooling, as the learner's: the squares of the feature maps (C, H, W) that the corners of `box` (top, left,
    bottom, right, in pixels) lie in, each square STRIDE by STRIDE pixels, kept inside the maps; then the highest
    number of each of k by k bins of those squares, for every channel: a tensor (C, k, k)."""
    top, left, bottom, right = box
    h, w = maps.shape[1], maps.shape[2]
    r0, r1 = min(max(int(top // 8), 0), h - 1), min(max(int(bottom // 8), 0), h - 1)
    c0, c1 = min(max(int(left // 8), 0), w - 1), min(max(int(right // 8), 0), w - 1)
    return F.adaptive_max_pool2d(maps[:, r0:r1 + 1, c0:c1 + 1], k)


def roi_align(maps, box, k: int = K, n: int = 2):
    """For the course's comparison, RoIAlign as Mask R-CNN does it, without rounding: in each of k by k bins of the box
    itself, n by n points at even distances, each the bilinear mix of the 4 squares around it (squares centred on their
    middle pixel), and the highest of them. A tensor (C, k, k)."""
    top, left, bottom, right = (float(v) for v in box)
    f = (torch.arange(k * n, dtype=torch.float32) + 0.5) / (k * n)   # the points, as shares of the box
    ys, xs = (top - 0.5) + f * (bottom - top + 1), (left - 0.5) + f * (right - left + 1)   # in pixels
    h, w = maps.shape[1], maps.shape[2]
    gy = ((ys - (STRIDE - 1) / 2) / STRIDE + 0.5) / h * 2 - 1   # square s has its middle at s, grid_sample wants -1 to 1
    gx = ((xs - (STRIDE - 1) / 2) / STRIDE + 0.5) / w * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1)[None]   # (1, kn, kn, 2), as (x, y)
    points = F.grid_sample(maps[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    return F.max_pool2d(points, n)


def proposals(out, anchors, n):
    """The proposals of one scene, as the learner's: the probability of an object of every anchor (softmax over its 2
    logits), its box from its 4 offsets, the BEST anchors with the highest probability, NMS at RPN_NMS, the first n that
    stay. A tensor (n, 4), the highest probability first; fewer rows when NMS keeps fewer. Softmax as the learner writes
    it, so that their code gives the same numbers as the course's run."""
    e = torch.exp(out[:, :2])
    score = e[:, 1] / e.sum(dim=1)
    boxes = boxes_from(anchors, out[:, 2:])
    best = score.argsort(descending=True)[:BEST]
    kept = scenes.nms(boxes[best], score[best], RPN_NMS)[:n]
    return boxes[best][kept]


def scored_proposals(out, anchors, n):
    """As proposals, with the probability of an object of each proposal, for the pictures."""
    e = torch.exp(out[:, :2])
    score = e[:, 1] / e.sum(dim=1)
    boxes = boxes_from(anchors, out[:, 2:])
    best = score.argsort(descending=True)[:BEST]
    kept = scenes.nms(boxes[best], score[best], RPN_NMS)[:n]
    return boxes[best][kept], score[best][kept]


class TwoStage(nn.Module):
    """The course's two-stage detector: the backbone of the one-stage detector, 64 feature maps of 16 by 16; the region
    proposal network, a convolution of 3 by 3 that gives each of the 3 anchors of a square 2 logits and 4 offsets; and
    the second stage, a linear layer of 128, ReLU, and a linear layer to 4 logits and 4 offsets for every proposal."""

    def __init__(self, k: int = K):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.rpn = nn.Conv2d(64, 3 * PER_RPN, 3, padding=1)
        self.second = nn.Sequential(nn.Linear(64 * k * k, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, PER))

    def forward(self, x):
        """The feature maps (N, 64, 16, 16) and the RPN's rows (N, 768, 6) of a batch of scenes."""
        maps = self.backbone(x)
        return maps, self.rpn(maps).permute(0, 2, 3, 1).reshape(len(x), -1, PER_RPN)


def as_boxes(boxes) -> torch.Tensor:
    """Boxes as a tensor of floats, one row (top, left, bottom, right) each: from a tensor, an array or a list."""
    if torch.is_tensor(boxes):
        return boxes.detach().float().reshape(-1, 4)
    return torch.as_tensor(np.asarray(boxes, dtype=np.float32)).reshape(-1, 4)


def pooled(maps, rois, pool_fn=roi_pool, k: int = K) -> torch.Tensor:
    """The pooled features of every (scene, box) in rois, the boxes of one scene after each other: (R, 64, k, k). The
    maps of a scene are taken once, so that training follows each scene back once, not once for every box."""
    out, last, mine = [], None, None
    for i, box in rois:
        if i != last:
            mine, last = maps[i], i
        out.append(pool_fn(mine, box, k))
    return torch.stack(out)


def second_targets(props, truth, kinds):
    """What every proposal learns in the second stage: the kind + 1 of the true box it overlaps most, from an IoU of
    FG_IOU on, else 0 for background; and the offsets from the proposal to that box (only the kinds use them)."""
    ious = torch.as_tensor(iou_matrix(props.numpy(), truth.numpy()))
    best, which = ious.max(dim=1)
    classes = torch.where(best >= FG_IOU, kinds[which] + 1, torch.zeros_like(which))
    return classes, offsets(props, truth[which])


def rpn_loss(rows, classes, t):
    """The loss of the RPN, the loss of the one-stage detector with 2 classes: the cross-entropy over the anchors that
    are not ignored, plus the mean absolute difference of the offsets over the anchors that learn a box."""
    kept, learn = classes >= 0, classes > 0
    return F.cross_entropy(rows[:, :, :2][kept], classes[kept]) + (rows[:, :, 2:][learn] - t[learn]).abs().mean()


def second_loss(out, classes, t):
    """The loss of the second stage: the cross-entropy over every proposal, plus the mean absolute difference of the
    offsets over the proposals that learn a kind."""
    learn = classes > 0
    return F.cross_entropy(out[:, :4], classes) + (out[learn, 4:] - t[learn]).abs().mean()


@functools.lru_cache(maxsize=1)
def rpn_targets() -> tuple:
    """For every training scene, what every anchor learns in the RPN: the classes of the one-stage detector, with every
    kind as class 1, and its offsets."""
    (C, T), _ = _ONE.course_targets()
    return C.clamp(max=1), T


def found_boxes(out, props, boxes_fn=boxes_from) -> list[tuple]:
    """The boxes of one scene from the second stage's rows `out` for its proposals `props`: (kind, score, box), the
    highest score first. The score is the highest probability of a kind; every proposal with a score of LOW or more
    becomes its box with boxes_fn, then NMS for each kind on its own."""
    p = out[:, :4].softmax(dim=1)
    score, kind = p[:, 1:].max(dim=1)
    keep = score >= LOW
    boxes = as_boxes(boxes_fn(props[keep], out[keep, 4:].clone())).tolist()
    s, k = score[keep].tolist(), kind[keep].tolist()
    found = []
    for kd in range(KINDS):
        mine = [i for i in range(len(boxes)) if k[i] == kd]
        for q in scenes.nms([boxes[i] for i in mine], [s[i] for i in mine], NMS_IOU):
            found.append((kd, s[mine[q]], tuple(boxes[mine[q]])))
    found.sort(key=lambda d: -d[1])
    return found


def detect(model, X, keep: int = TEST_KEEP, pool_fn=roi_pool, prop_fn=proposals, boxes_fn=boxes_from) -> list[tuple]:
    """For every scene of X (bytes): its boxes, as found_boxes, and its proposals; in eval mode, without a graph."""
    model.eval()
    a = anchors()
    out_all = []
    with torch.no_grad():
        for i in range(0, len(X), 50):
            maps, rows = model(X[i:i + 50].float() / 255)
            for j in range(len(maps)):
                props = as_boxes(prop_fn(rows[j], a.clone(), keep))
                if len(props) == 0:
                    out_all.append(([], props))
                    continue
                out = model.second(pooled(maps, [(j, tuple(b)) for b in props.tolist()], pool_fn).reshape(len(props), -1))
                out_all.append((found_boxes(out, props, boxes_fn), props))
    model.train()
    return out_all


def shown(found: list) -> list[tuple]:
    """The boxes with a score of SCORE or more, rounded as the lesson writes them: (kind, score, box)."""
    return [(k, round(s, 3), tuple(round(v, 1) for v in b)) for k, s, b in found if s >= SCORE]


def summary(results: list, ss) -> dict:
    """The mAP at 0.5 and its AP per kind, the mAP at 0.75, the shapes found, the false boxes and the shapes with a
    proposal, of the boxes and proposals of every scene of ss."""
    found = [f for f, _ in results]
    m, aps = mean_ap(found, ss)
    m75, _ = mean_ap(found, ss, 0.75)
    right = false = covered = 0
    for (f, props), s in zip(results, ss):
        h = hits([d for d in f if d[1] >= SCORE], s)
        right, false = right + sum(h), false + len(h) - sum(h)
        boxes = props.tolist()
        covered += sum(any(iou(b, shape.box) > AP_IOU for b in boxes) for shape in s.shapes)
    return {"map": round(m, 4), "aps": [round(a, 4) for a in aps], "map75": round(m75, 4), "found": right,
            "false": false, "covered": covered, "shapes": sum(len(s.shapes) for s in ss)}


def train(model, seed: int = 0, epochs: int = EPOCHS, heard=None, pool_fn=roi_pool, prop_fn=proposals) -> list[dict]:
    """Train `model` on the training scenes, as the course measured: for every batch, the RPN's loss, and for every scene
    its proposals (prop_fn, TRAIN_KEEP of them, with the true boxes added), their pooled features (pool_fn) and the loss
    of the second stage on them; the sum of the two. The proposals count as given numbers. `heard` hears every 10
    batches how far an epoch is, and after every epoch the loss of each stage and the boxes on the SHOWN scenes."""
    ss, X = scene_set("train")
    C, T = rpn_targets()
    truths = [boxes_and_labels(s) for s in ss]
    a = anchors()
    n = len(X)
    per = math.ceil(n / BATCH)
    total = epochs * per
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, per, total))
    order_rng = torch.Generator().manual_seed(seed + 1)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=order_rng)
        run_rpn = run_second = 0.0
        for b, i in enumerate(range(0, n, BATCH), 1):
            idx = order[i:i + BATCH]
            maps, rows = model(X[idx].float() / 255)
            first = rpn_loss(rows, C[idx], T[idx])
            rois, classes, ts = [], [], []
            for j, s_i in enumerate(idx.tolist()):
                truth, kinds = truths[s_i]
                props = torch.cat([as_boxes(prop_fn(rows[j].detach(), a.clone(), TRAIN_KEEP)), truth])
                c, t = second_targets(props, truth, kinds)
                rois += [(j, tuple(p)) for p in props.tolist()]
                classes.append(c)
                ts.append(t)
            out = model.second(pooled(maps, rois, pool_fn).reshape(len(rois), -1))
            second = second_loss(out, torch.cat(classes), torch.cat(ts))
            loss = first + second
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_rpn += first.item()
            run_second += second.item()
            if heard is not None and b % 10 == 0:
                heard({"epoch": epoch, "epochs": epochs, "batch": b, "batches": per})
        history.append({"epoch": epoch, "rpn_loss": run_rpn / per, "second_loss": run_second / per})
        if heard is not None:
            heard({"epoch": epoch, "epochs": epochs, "rpn_loss": run_rpn / per, "second_loss": run_second / per,
                   "model": model})
    return history


def train_course(seed: int = 0, pool_fn=roi_pool, heard=None) -> tuple:
    """The course's two-stage detector, trained with `seed` on one thread: (model, history)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = TwoStage()
        history = train(model, seed, EPOCHS, heard, pool_fn=pool_fn)
    finally:
        torch.set_num_threads(threads)
    return model, history


def shown_moment(model, pool_fn=roi_pool, prop_fn=proposals, boxes_fn=boxes_from) -> list[list]:
    """The boxes of 0.5 or more on each of the SHOWN scenes, for the rewind."""
    ss, X = scene_set("test")
    idx = torch.tensor([TEST.index(seed) for seed in SHOWN])
    return [shown(f) for f, _ in detect(model, X[idx], TEST_KEEP, pool_fn, prop_fn, boxes_fn)]


def measure_seed(seed: int, pool_fn=roi_pool) -> dict:
    """Train with `seed` and measure everything TRAINED holds for one seed; for seed 0 also the scene, the curves and
    every number of proposals kept."""
    model, history = train_course(seed, pool_fn)
    ss, Xt = scene_set("test")
    out = summary(detect(model, Xt, TEST_KEEP, pool_fn), ss)
    out["curves"] = {"rpn": [round(h["rpn_loss"], 4) for h in history], "second": [round(h["second_loss"], 4) for h in history]}
    out["keeps"] = {}
    for n in KEEPS:
        results = detect(model, Xt, n, pool_fn)
        out["keeps"][n] = summary(results, ss) | {"boxes": shown(results[TEST.index(SEED)][0])}
    model.eval()
    with torch.no_grad():
        x = torch.tensor(example().image).permute(2, 0, 1).unsqueeze(0).float() / 255
        maps, rows = model(x)
        props, score = scored_proposals(rows[0], anchors(), TEST_KEEP)
    out["scene"] = {"proposals": [(round(float(s), 3), tuple(round(v, 1) for v in b)) for b, s in zip(props.tolist(), score.tolist())],
                    "after": out["keeps"][TEST_KEEP]["boxes"]}
    return out


# --- the worked example ---------------------------------------------------------------------------------------------

def _bins(n: int, k: int = K) -> list[tuple[int, int]]:
    """The rows (or columns) of each of k bins of n rows, as RoI pooling cuts them: from floor(i n / k) up to, not
    including, ceil((i + 1) n / k)."""
    return [(math.floor(i * n / k), math.ceil((i + 1) * n / k)) for i in range(k)]


def worked() -> dict:
    """The worked example's numbers, as its cell computes them, with an untrained network after torch.manual_seed(0)."""
    s = example()
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    a = anchors()
    torch.manual_seed(0)
    net = TwoStage()
    with torch.no_grad():
        maps = net.backbone(x)[0]
        rows = net.rpn(maps.unsqueeze(0)).permute(0, 2, 3, 1).reshape(768, 6)
        e = torch.exp(rows[:, :2])
        score = e[:, 1] / e.sum(dim=1)
        boxes = boxes_from(a, rows[:, 2:])
        best = score.argsort(descending=True)[:BEST]
        kept = scenes.nms(boxes[best], score[best], RPN_NMS)[:TEST_KEEP]
        props = boxes[best][kept]
        box = s.shapes[POOLED].box
        top, left, bottom, right = box
        r0, c0, r1, c1 = int(top // 8), int(left // 8), int(bottom // 8), int(right // 8)
        region = maps[:, r0:r1 + 1, c0:c1 + 1]
        mine = F.adaptive_max_pool2d(region, K)
        all_pooled = torch.stack([roi_pool(maps, b, K) for b in props.tolist()])
        out = net.second(all_pooled.reshape(len(props), -1))
    return {"maps": tuple(maps.shape), "rows": tuple(rows.shape), "kept_after_nms": len(scenes.nms(boxes[best], score[best], RPN_NMS)),
            "proposals": len(props), "first": tuple(round(v, 1) for v in props[0].tolist()),
            "score": (round(float(score.max()), 4), round(float(score.min()), 4)),
            "box": tuple(box), "squares": (r0, c0, r1, c1), "region": tuple(region.shape),
            "bins": (_bins(region.shape[1]), _bins(region.shape[2])),
            "pooled": tuple(round(v, 4) for v in mine[0].reshape(-1).tolist()),
            "pooled_shape": tuple(all_pooled.shape), "out": tuple(out.shape),
            "params": sum(p.numel() for p in net.parameters()),
            "second_params": sum(p.numel() for p in net.second.parameters()),
            "rpn_params": sum(p.numel() for p in net.rpn.parameters())}


def measure() -> dict:
    return worked()


# --- pictures ---------------------------------------------------------------------------------------------------------

CROP = _ONE.CROP


def _thin_box(ax, b):
    """A proposal: a thin dashed line in the ink of the page, so that 32 of them stay readable."""
    from everything_ai import plots
    _ONE._draw_box(ax, b, plots.INK_2, (0, (2, 1.4)), width=0.9, halo=1.8)


def _proposals_svg(s, props, title, figsize=(3.9, 4.05), crop=CROP, size=10):
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    fig, ax = plt.subplots(figsize=figsize)
    _ONE._photo_axes(ax, s.image, title, crop=crop, size=size)
    for shape in s.shapes:
        _ONE._truth_box(ax, shape.box)
    for b in props:
        _thin_box(ax, b)
    svg = plots.render(fig)
    plt.close(fig)
    return svg


def stages_picture(words: dict[str, str]) -> str:
    """Two pictures of the scene of the worked example and the course's trained two-stage detector (seed 0): its 32
    proposals; and the boxes of 0.5 or more that the second stage makes of them, after NMS, with their kind and score."""
    s = example()
    props = [b for _, b in TRAINED["scene"]["proposals"]]
    after = TRAINED["scene"]["after"]
    left = _proposals_svg(s, props, words["first"].format(n=len(props)))
    right = _ONE._scene_svg(s, after, words["second"].format(n=len(after)), names=words)
    return _ONE._side_by_side([(left, 300), (right, 300)], words["alt"])


def roi_example() -> dict:
    """For the picture of RoI pooling: the first proposal of the course's trained detector (seed 0) on the scene of the
    worked example, on the large triangle, its squares, and its bins."""
    box = TRAINED["scene"]["proposals"][0][1]
    top, left, bottom, right = box
    r0, c0, r1, c1 = int(top // 8), int(left // 8), int(bottom // 8), int(right // 8)
    return {"box": box, "squares": (r0, c0, r1, c1), "bins": (_bins(r1 - r0 + 1), _bins(c1 - c0 + 1))}


def roi_picture(words: dict[str, str]) -> str:
    """RoI pooling of one proposal, on the scene of the worked example: the squares of 8 by 8 pixels, the proposal
    (dashed), the squares its corners lie in (solid), and those cut into 4 by 4 bins (thin lines, where bins meet)."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    ex = roi_example()
    r0, c0, r1, c1 = ex["squares"]
    s = example()
    crop = (max(r0 * 8 - 12, 0), max(c0 * 8 - 12, 0), min(r1 * 8 + 19, 127), min(c1 * 8 + 19, 127))
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    _ONE._photo_axes(ax, s.image, words["title"], crop=crop, size=10)
    for k in range(17):   # the squares of the feature maps, 8 by 8 pixels
        ax.axhline(k * 8 - 0.5, color=plots.PAGE, linewidth=0.6, alpha=0.7)
        ax.axvline(k * 8 - 0.5, color=plots.PAGE, linewidth=0.6, alpha=0.7)
    rows, cols = ex["bins"]
    for colour, width in ((plots.PAGE, 2.6), (plots.INK, 1.0)):   # ink with an edge of the page: seen on the photo in light and dark
        for a, _ in rows[1:]:   # where a bin starts, inside the squares of the proposal
            ax.plot([c0 * 8 - 0.5, (c1 + 1) * 8 - 0.5], [(r0 + a) * 8 - 0.5] * 2, color=colour, linewidth=width)
        for a, _ in cols[1:]:
            ax.plot([(c0 + a) * 8 - 0.5] * 2, [r0 * 8 - 0.5, (r1 + 1) * 8 - 0.5], color=colour, linewidth=width)
    _ONE._draw_box(ax, (r0 * 8, c0 * 8, (r1 + 1) * 8 - 1, (c1 + 1) * 8 - 1), plots.INK, "-", width=2.0, halo=4.0)   # the squares
    _ONE._draw_box(ax, ex["box"], plots.PRED, (0, (3, 1.6)))
    svg = plots.render(fig)
    plt.close(fig)
    return f'<div role="img" aria-label="{_escape(words["alt"])}">{svg}</div>'


def _escape(text: str) -> str:
    import html
    return html.escape(text, quote=True)


def keep_frame(n: int, words: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the boxes of 0.5 or more after NMS on the scene of the worked example, when the course's detector
    keeps n proposals to detect; next to it its mAP at an IoU of 0.5 and of 0.75 for every number of proposals."""
    stop = TRAINED["keeps"][n]
    scene = _ONE._scene_svg(example(), stop["boxes"], words["title"].format(n=n, boxes=len(stop["boxes"])),
                            names=words, figsize=(2.9, 3.0), size=9)
    xs = list(range(len(KEEPS)))   # each number of proposals twice the one before: equal steps
    p = chart.Panel((-0.4, len(KEEPS) - 0.6), (0, 1.05), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], title=words["chart"],
                    xlabel=words["keep"], ylabel="mAP", xticks=xs, xtick_labels=[str(v) for v in KEEPS])
    p.line(xs, [TRAINED["keeps"][v]["map"] for v in KEEPS], "ink", 2.2)
    p.line(xs, [TRAINED["keeps"][v]["map75"] for v in KEEPS], "ink", 2.2, dashed=True)
    i = KEEPS.index(n)
    p.points([i, i], [stop["map"], stop["map75"]], "ink", r=4.5)
    p.text(-0.3, 0.96, words["at50"], "ink", "start")   # top left, above the lines where they start low
    p.text(-0.3, 0.84, words["at75"], "ink", "start")
    plot = chart.figure([p], alt=words["chart_alt"], width=276, height=236)   # 11 px text in a play box on a phone
    if words["point"] == ",":
        plot = chart.with_decimal_comma(plot)
    return _ONE._side_by_side([(scene, 250), (plot, 276)], words["alt"].format(n=n, boxes=len(stop["boxes"]))), stop


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss of each stage after every epoch of the demo's run: the RPN's solid, the second stage's dashed. With
    `at`, a line at that epoch, for rewinding."""
    first, second = curves.get("rpn", []), curves.get("second", [])
    mine = [v for v in first + second if np.isfinite(v)]
    top = max([0.2] + [1.1 * v for v in mine])
    p = chart.Panel((0.5, max(len(first), len(second), 1) + 0.5), (0, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for ys, dashed, name, dy in ((second, True, labels["second"], 0.07), (first, False, labels["rpn"], 0.07)):
        if ys:
            p.line(list(range(1, len(ys) + 1)), ys, "loss", 2.2, dashed=dashed)
            p.text(len(ys) - 0.3 if len(ys) > 1 else 1, ys[-1] + dy * top, name, "loss", "end")   # above the line at the end
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=276, height=200)   # 11 px text in a play box on a phone
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_picture(boxes: list, words: dict) -> str:
    return _ONE.rewind_picture(boxes, words)


# --- exercise 2: roi_pool ---------------------------------------------------------------------------------------------

_R_HEAD = ("import torch.nn.functional as F\n\n\ndef roi_pool(maps, box, k):\n"
           "    # maps: the feature maps of one scene, shape (C, H, W), every square 8 by 8 pixels\n"
           "    # box: (top, left, bottom, right) in pixels; it may stick out of the scene\n")
_R_START = (_R_HEAD + "    # give back the highest number of each of k by k bins of the squares its corners lie in: (C, k, k)\n"
            "    top, left, bottom, right = box\n    ...\n")
_R_UNPACK = "    top, left, bottom, right = box\n"
_R_SIZE = "    h, w = maps.shape[1], maps.shape[2]\n"
_R_ROWS = "    r0, r1 = min(max(int(top // 8), 0), h - 1), min(max(int(bottom // 8), 0), h - 1)\n"
_R_COLS = "    c0, c1 = min(max(int(left // 8), 0), w - 1), min(max(int(right // 8), 0), w - 1)\n"
_R_POOL = "    return F.adaptive_max_pool2d(maps[:, r0:r1 + 1, c0:c1 + 1], k)\n"


def _r(size=_R_SIZE, rows=_R_ROWS, cols=_R_COLS, pool=_R_POOL) -> str:
    return _R_HEAD + "    # give back the highest number of each of k by k bins of the squares its corners lie in: (C, k, k)\n" + _R_UNPACK + size + rows + cols + pool


_R = _r()
_R_MAPS = "import torch\n\nmaps = torch.randperm({n}, generator=torch.Generator().manual_seed({seed})).reshape({shape}).float()\n"
_R_SCENARIOS = [
    # 6 by 6 squares, a box over rows 0 to 3 and columns 1 to 5: 2 by 2 bins, the columns of the bins overlap
    _R_MAPS.format(n=2 * 36, seed=0, shape="2, 6, 6") + "roi_pool(maps, (5.5, 9.0, 30.2, 44.0), 2).tolist()",
    # a box that sticks out of the scene, above and to the right
    _R_MAPS.format(n=36, seed=1, shape="1, 6, 6") + "roi_pool(maps, (-6.0, 30.0, 20.0, 60.0), 2).tolist()",
    # a box inside one square: every bin is that square
    _R_MAPS.format(n=36, seed=2, shape="1, 6, 6") + "roi_pool(maps, (17.0, 17.5, 22.0, 23.0), 2).tolist()",
    # the maps of a scene of 128 by 128 pixels, the circle of the worked example, 4 by 4 bins
    _R_MAPS.format(n=3 * 256, seed=3, shape="3, 16, 16") + "roi_pool(maps, (51.0, 47.0, 82.0, 78.0), 4)[1].tolist()",
    # maps of another size, and a box below them
    _R_MAPS.format(n=3 * 12 * 10, seed=4, shape="3, 12, 10") + "roi_pool(maps, (100.5, 70.0, 130.0, 90.0), 2).tolist()",
]
_R_WRONG = {
    "no-plus-one": _r(pool="    return F.adaptive_max_pool2d(maps[:, r0:r1, c0:c1], k)\n"),
    "round": _r(rows="    r0, r1 = min(max(round(top / 8), 0), h - 1), min(max(round(bottom / 8), 0), h - 1)\n",
                cols="    c0, c1 = min(max(round(left / 8), 0), w - 1), min(max(round(right / 8), 0), w - 1)\n"),
    "swapped": _r(pool="    return F.adaptive_max_pool2d(maps[:, c0:c1 + 1, r0:r1 + 1], k)\n"),
    "average": _r(pool="    return F.adaptive_avg_pool2d(maps[:, r0:r1 + 1, c0:c1 + 1], k)\n"),
    "pixels": _r(rows="    r0, r1 = min(max(int(top), 0), h - 1), min(max(int(bottom), 0), h - 1)\n",
                 cols="    c0, c1 = min(max(int(left), 0), w - 1), min(max(int(right), 0), w - 1)\n"),
    "sixteen": _r(rows="    r0, r1 = min(max(int(top // 16), 0), h - 1), min(max(int(bottom // 16), 0), h - 1)\n",
                  cols="    c0, c1 = min(max(int(left // 16), 0), w - 1), min(max(int(right // 16), 0), w - 1)\n"),
}

_DOTS = _ONE._DOTS
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close, _raised_in = _ONE._close, _ONE._raised_in


def _empty_region(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "non-zero size" in str(e)


_R_MISTAKES = [
    Mistake("not-whole", error=lambda e: isinstance(e, TypeError) and re.search(r"slice indices must be integers|integer tensors of a single element|indices must be integers", str(e)) is not None,
            en="Python stops when your code takes the squares out of `maps`: a square must be a whole number, and `top // 8` of a box of floats gives a float, such as `5.0`. Make it a whole number with `int(...)`: `int(top // 8)`.",
            nl="Python stopt als je code de vakjes uit `maps` haalt: een vakje moet een geheel getal zijn, en `top // 8` van een box van floats geeft een float, zoals `5.0`. Maak er een geheel getal van met `int(...)`: `int(top // 8)`."),
    Mistake("empty", error=_empty_region,
            en="Python stops in `F.adaptive_max_pool2d`: the squares you give it are empty. Is the last square in it, `r0:r1 + 1`? A slice stops before its end. And does a box that sticks out of the scene above or to the left still start at square 0, and one below the scene at the last square? `min(max(..., 0), h - 1)` keeps a square inside the maps.",
            nl="Python stopt in `F.adaptive_max_pool2d`: de vakjes die je hem geeft, zijn leeg. Zit het laatste vakje erin, `r0:r1 + 1`? Een slice stopt vóór zijn eind. En begint een box die boven of links uit de scène steekt nog bij vakje 0, en een box onder de scène bij het laatste vakje? `min(max(..., 0), h - 1)` houdt een vakje binnen de maps."),
    Mistake("no-plus-one", wrong=_R_WRONG["no-plus-one"],
            en=_S_EN + ": the last row and the last column of squares are missing. A slice stops before its end: `maps[:, r0:r1 + 1, c0:c1 + 1]` takes the squares from `r0` to `r1`, both included.",
            nl=_S_NL + ": de laatste rij en de laatste kolom vakjes ontbreken. Een slice stopt vóór zijn eind: `maps[:, r0:r1 + 1, c0:c1 + 1]` neemt de vakjes van `r0` tot en met `r1`."),
    Mistake("round", wrong=_R_WRONG["round"],
            en=_S_EN + ": you round to the nearest square, and a corner of the box can lie in the square before it. The square a pixel lies in is `int(pixel // 8)`: pixels 0 to 7 lie in square 0, 8 to 15 in square 1.",
            nl=_S_NL + ": je rondt af op het dichtstbijzijnde vakje, en een hoek van de box kan in het vakje ervoor liggen. Het vakje waar een pixel in ligt, is `int(pixel // 8)`: pixels 0 tot en met 7 liggen in vakje 0, 8 tot en met 15 in vakje 1."),
    Mistake("swapped", wrong=_R_WRONG["swapped"],
            en=_S_EN + ": the rows and the columns are swapped. The maps have the rows first, from `top` and `bottom`, then the columns, from `left` and `right`: `maps[:, r0:r1 + 1, c0:c1 + 1]`.",
            nl=_S_NL + ": de rijen en de kolommen zijn verwisseld. De maps hebben de rijen eerst, uit `top` en `bottom`, en dan de kolommen, uit `left` en `right`: `maps[:, r0:r1 + 1, c0:c1 + 1]`."),
    Mistake("average", wrong=_R_WRONG["average"],
            en=_S_EN + ": you take the mean of every bin. RoI pooling keeps the highest number of every bin, as max pooling does: `F.adaptive_max_pool2d`.",
            nl=_S_NL + ": je neemt het gemiddelde van elke bin. RoI pooling houdt het hoogste getal van elke bin, zoals max pooling: `F.adaptive_max_pool2d`."),
    Mistake("pixels", wrong=_R_WRONG["pixels"],
            en=_S_EN + ": you use the pixels as squares. Every square of the maps stands for 8 by 8 pixels: the square of a pixel is `int(pixel // 8)`.",
            nl=_S_NL + ": je gebruikt de pixels als vakjes. Elk vakje van de maps staat voor 8 bij 8 pixels: het vakje van een pixel is `int(pixel // 8)`."),
    Mistake("sixteen", wrong=_R_WRONG["sixteen"],
            en=_S_EN + ": you divide by 16, as for the maps of VGG16 in Fast R-CNN. Here every square stands for 8 by 8 pixels: `int(pixel // 8)`.",
            nl=_S_NL + ": je deelt door 16, zoals voor de maps van VGG16 in Fast R-CNN. Hier staat elk vakje voor 8 bij 8 pixels: `int(pixel // 8)`."),
]

# --- exercise 3: proposals --------------------------------------------------------------------------------------------

_P_HEAD = ("import torch\nfrom cnn.offsets import boxes_from\nfrom everything_ai.scenes import nms\n\n\ndef proposals(out, anchors, n):\n"
           "    # out: the RPN's rows of one scene, shape (A, 6): 2 logits (background, object), 4 offsets\n"
           "    # anchors: (A, 4); give back the boxes of the first n proposals: at most n rows of 4\n")
_P_START = _P_HEAD + "    ...\n"
_P_SCORE = "    e = torch.exp(out[:, :2])\n    score = e[:, 1] / e.sum(dim=1)\n"
_P_BOXES = "    boxes = boxes_from(anchors, out[:, 2:])\n"
_P_BEST = "    best = score.argsort(descending=True)[:100]\n"
_P_NMS = "    kept = nms(boxes[best], score[best], 0.7)[:n]\n"
_P_RETURN = "    return boxes[best][kept]\n"


def _p(score=_P_SCORE, boxes=_P_BOXES, best=_P_BEST, nms=_P_NMS, ret=_P_RETURN) -> str:
    return _P_HEAD + score + boxes + best + nms + ret


_P = _p()
_P_SIX = ("import torch\n\n"
          "anchors = torch.tensor([[0.0, 0.0, 15.0, 15.0], [0.0, 2.0, 15.0, 17.0], [20.0, 20.0, 35.0, 35.0],\n"
          "                        [40.0, 0.0, 55.0, 15.0], [40.0, 40.0, 55.0, 55.0], [40.0, 44.0, 55.0, 59.0]])\n"
          "out = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],\n"
          "                    [0.0, 1.5, 0.25, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]])\n")
_P_SCENE = ("import torch\nfrom everything_ai.scenes import anchor_boxes\n\n"
            "anchors = torch.tensor(anchor_boxes()).float()\nout = torch.randn(768, 6, generator=torch.Generator().manual_seed({seed}))\n")
_P_SCENARIOS = [
    # six anchors: 1 overlaps 0 by 0.78 and goes, 4 and 5 overlap by 0.6 and both stay; 3 moves 4 pixels down
    _P_SIX + "proposals(out, anchors, 6).tolist()",
    # the same six, the first 2
    _P_SIX + "proposals(out, anchors, 2).tolist()",
    # 768 anchors with random rows, as an RPN that has not learned yet
    _P_SCENE.format(seed=0) + "proposals(out, anchors, 5).tolist()",
    _P_SCENE.format(seed=1) + "proposals(out, anchors, 12)[-3:].tolist()",
]
_P_WRONG = {
    "logit": _p(score="    score = out[:, 1]\n"),
    "background": _p(score="    e = torch.exp(out[:, :2])\n    score = e[:, 0] / e.sum(dim=1)\n"),
    "ascending": _p(best="    best = score.argsort()[:100]\n"),
    "kept-index": _p(ret="    return boxes[kept]\n"),
    "no-nms": _p(nms="    kept = list(range(len(best)))[:n]\n"),
    "half": _p(nms="    kept = nms(boxes[best], score[best], 0.5)[:n]\n"),
    "columns": _p(boxes="    boxes = boxes_from(anchors, out[:, :4])\n"),
}
_P_MISTAKES = [
    Mistake("pair", error=lambda e: isinstance(e, AttributeError) and "tolist" in str(e),
            en="The check uses what your function gives back as one tensor of boxes, and your function gives back something else, such as the boxes and the scores together. Give back only the boxes: `return boxes[best][kept]`.",
            nl="De check gebruikt wat je functie teruggeeft als één tensor van boxes, en je functie geeft iets anders terug, zoals de boxes en de scores samen. Geef alleen de boxes terug: `return boxes[best][kept]`."),
    Mistake("logit", wrong=_P_WRONG["logit"],
            en=_S_EN + ": you sort the anchors by their logit of an object. How sure the RPN is depends on both logits: a logit of 4 for an object counts for little when background has 3. Softmax over the two gives the probability: `e = torch.exp(out[:, :2])`, then `e[:, 1] / e.sum(dim=1)`.",
            nl=_S_NL + ": je sorteert de anchors op hun logit van een voorwerp. Hoe zeker het RPN is, hangt van beide logits af: een logit van 4 voor een voorwerp zegt weinig als achtergrond er 3 heeft. Softmax over de twee geeft de kans: `e = torch.exp(out[:, :2])`, en dan `e[:, 1] / e.sum(dim=1)`."),
    Mistake("background", wrong=_P_WRONG["background"],
            en=_S_EN + ": you take the probability of background, column 0. The probability of an object is column 1: `e[:, 1] / e.sum(dim=1)`.",
            nl=_S_NL + ": je neemt de kans op achtergrond, kolom 0. De kans op een voorwerp is kolom 1: `e[:, 1] / e.sum(dim=1)`."),
    Mistake("ascending", wrong=_P_WRONG["ascending"],
            en=_S_EN + ": `argsort` puts the lowest score first, so you keep the 100 anchors that are surest of background. `argsort(descending=True)` puts the highest first.",
            nl=_S_NL + ": `argsort` zet de laagste score vooraan, dus houd je de 100 anchors die het zekerst achtergrond zijn. `argsort(descending=True)` zet de hoogste vooraan."),
    Mistake("kept-index", wrong=_P_WRONG["kept-index"],
            en=_S_EN + ": `nms` gives the places of the boxes it keeps in what you gave it, `boxes[best]`, not in `boxes`. Take them from there: `boxes[best][kept]`.",
            nl=_S_NL + ": `nms` geeft de plaatsen van de boxes die hij houdt in wat je hem gaf, `boxes[best]`, niet in `boxes`. Haal ze daar ook uit: `boxes[best][kept]`."),
    Mistake("no-nms", wrong=_P_WRONG["no-nms"],
            en=_S_EN + ": without NMS, the first proposals are near copies of each other, around the same shape. `nms(boxes[best], score[best], 0.7)` drops every box that overlaps a surer one by more than 0.7.",
            nl=_S_NL + ": zonder NMS zijn de eerste proposals bijna kopieën van elkaar, rond dezelfde vorm. `nms(boxes[best], score[best], 0.7)` laat elke box vallen die een zekerdere box met een IoU boven 0,7 overlapt."),
    Mistake("half", wrong=_P_WRONG["half"],
            en=_S_EN + ": your NMS drops boxes from an overlap of 0.5. For proposals, it is 0.7, as in Faster R-CNN: that leaves a few different proposals around every shape for the second stage, and the NMS at the end removes the copies.",
            nl=_S_NL + ": je NMS laat boxes vallen vanaf een overlap van 0,5. Voor proposals is het 0,7, zoals in Faster R-CNN: zo blijven er rond elke vorm een paar verschillende proposals over voor de tweede fase, en de NMS aan het eind haalt de kopieën weg."),
    Mistake("columns", wrong=_P_WRONG["columns"],
            en=_S_EN + ": your offsets are the first 4 numbers of a row, and two of those are the logits. The offsets are the last 4: `out[:, 2:]`.",
            nl=_S_NL + ": je offsets zijn de eerste 4 getallen van een rij, en twee daarvan zijn de logits. De offsets zijn de laatste 4: `out[:, 2:]`."),
]

EXERCISES = {
    "roi-pool": ScenarioBox(
        name="roi-pool",
        picture=yours.arrays(call=0, result="tuple(roi_pool(maps, (5.5, 9.0, 30.2, 44.0), 2))", shade=True),
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "roi_pool"),
        first=[_DOTS],
        mistakes=_R_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the squares of the box's corners, `int(pixel // 8)`, kept from 0 to the last square of the maps; then the maps of those squares, both ends included, into `F.adaptive_max_pool2d` with `k`.",
                      "nl": "Vervang de `...` door je code: de vakjes van de hoeken van de box, `int(pixel // 8)`, binnen 0 en het laatste vakje van de maps gehouden; en dan de maps van die vakjes, beide einden erbij, in `F.adaptive_max_pool2d` met `k`."},
            "wrong": {"en": _S_EN + ". Do the squares run from the square of `top` to the square of `bottom`, and from `left` to `right`, both ends included, each `int(pixel // 8)`? And does every bin keep its highest number?",
                      "nl": _S_NL + ". Lopen de vakjes van het vakje van `top` tot en met dat van `bottom`, en van `left` tot en met `right`, elk `int(pixel // 8)`? En houdt elke bin zijn hoogste getal?"},
            "almost": {"en": "Right for the first box. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for maps of any size, with every square kept from 0 to the last square of the maps, `h - 1` and `w - 1`?",
                       "nl": "Goed voor de eerste box. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor maps van elke grootte, met elk vakje binnen 0 en het laatste vakje van de maps, `h - 1` en `w - 1`?"},
            "correct": {"en": "Your `roi_pool` is right: every box, of any size, becomes k by k numbers per channel. It is now in your project, as `{saved}`.",
                        "nl": "Je `roi_pool` klopt: elke box, van elke grootte, wordt k bij k getallen per channel. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `roi_pool` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `roi_pool` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "proposals": ScenarioBox(
        name="proposals",
        start=_P_START,
        solution=_P,
        scenarios=_P_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "proposals"),
        first=[_DOTS],
        mistakes=_P_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the probability of an object of every anchor, its box with `boxes_from`, the 100 anchors with the highest probability, `nms` on their boxes at 0.7, and the boxes of the first `n` it keeps.",
                      "nl": "Vervang de `...` door je code: de kans op een voorwerp van elke anchor, zijn box met `boxes_from`, de 100 anchors met de hoogste kans, `nms` op hun boxes bij 0,7, en de boxes van de eerste `n` die hij houdt."},
            "wrong": {"en": _S_EN + ". Is the score the probability of an object, from softmax over the 2 logits? Do you take the 100 highest, run `nms` on their boxes at 0.7, and give back the boxes of the first `n` it keeps, in its order?",
                      "nl": _S_NL + ". Is de score de kans op een voorwerp, uit softmax over de 2 logits? Neem je de 100 hoogste, laat je `nms` op hun boxes los bij 0,7, en geef je de boxes van de eerste `n` die hij houdt terug, in zijn volgorde?"},
            "almost": {"en": "Right for the six anchors with an `n` of 6. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of anchors and any `n`?",
                       "nl": "Goed voor de zes anchors met een `n` van 6. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal anchors en elke `n`?"},
            "correct": {"en": "Your `proposals` is right: the surest boxes, without near copies. It is now in your project, as `{saved}`.",
                        "nl": "Je `proposals` klopt: de zekerste boxes, zonder bijna-kopieën. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `proposals` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `proposals` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "fast" and "learns" are choices, "why-second" the open question.
REQUIRED = ["fast", "roi-pool", "proposals", "learns", "why-second"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("roi-pool", _R_START, "empty", None),
    ("roi-pool", _R, "correct", None),
    # the bins by hand, as in the worked example
    ("roi-pool", _r(pool="    region = maps[:, r0:r1 + 1, c0:c1 + 1]\n    import math\n    import torch\n"
                         "    out = torch.zeros(maps.shape[0], k, k)\n    for i in range(k):\n        for j in range(k):\n"
                         "            a, b = math.floor(i * region.shape[1] / k), math.ceil((i + 1) * region.shape[1] / k)\n"
                         "            c, d = math.floor(j * region.shape[2] / k), math.ceil((j + 1) * region.shape[2] / k)\n"
                         "            out[:, i, j] = region[:, a:b, c:d].reshape(maps.shape[0], -1).max(dim=1).values\n"
                         "    return out\n"), "correct", None),
    # a small helper for the square of a pixel
    ("roi-pool", _r(size=_R_SIZE + "    def square(p, last):\n        return min(max(int(p // 8), 0), last)\n",
                    rows="    r0, r1 = square(top, h - 1), square(bottom, h - 1)\n",
                    cols="    c0, c1 = square(left, w - 1), square(right, w - 1)\n"), "correct", None),
    ("roi-pool", _r(rows="    r0, r1 = int(top // 8), int(bottom // 8)\n", cols="    c0, c1 = int(left // 8), int(right // 8)\n"), "error", "empty"),
    ("roi-pool", _r(rows="    r0, r1 = min(max(top // 8, 0), h - 1), min(max(bottom // 8, 0), h - 1)\n"), "error", "not-whole"),
    ("roi-pool", _r(pool="    region = ...\n    return F.adaptive_max_pool2d(region, k)\n"), "wrong", "dots-left"),
    # the last square of a scene of 128 pixels, 15, as a fixed number: right for maps of 16 by 16, and empty for a box
    # below maps of 12 rows
    ("roi-pool", _r(size="", rows="    r0, r1 = min(max(int(top // 8), 0), 15), min(max(int(bottom // 8), 0), 15)\n",
                    cols="    c0, c1 = min(max(int(left // 8), 0), 15), min(max(int(right // 8), 0), 15)\n"), "error", "empty"),
] + [("roi-pool", code, "wrong", key) for key, code in _R_WRONG.items()] + [
    ("proposals", _P_START, "empty", None),
    ("proposals", _P, "correct", None),
    # NMS on every anchor, without the 100 best first: the same first n, more slowly
    ("proposals", _p(best="", nms="    kept = nms(boxes, score, 0.7)[:n]\n", ret="    return boxes[kept]\n"), "correct", None),
    # softmax in one step, and the probability of an object as a sigmoid of the difference of the logits
    ("proposals", _p(score="    score = out[:, :2].softmax(dim=1)[:, 1]\n"), "correct", None),
    ("proposals", _p(score="    score = torch.sigmoid(out[:, 1] - out[:, 0])\n"), "correct", None),
    ("proposals", _p(ret="    return boxes[best][kept], score[best][kept]\n"), "error", "pair"),
    ("proposals", _p(nms="    kept = ...\n"), "wrong", "dots-left"),
    # always 32 proposals, as the demo keeps: right for 6 anchors, where NMS keeps 5
    ("proposals", _p(nms="    kept = nms(boxes[best], score[best], 0.7)[:32]\n"), "almost", None),
] + [("proposals", code, "wrong", key) for key, code in _P_WRONG.items()]


# --- the demo: the course's two-stage detector with the learner's roi_pool and proposals ------------------------------

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
    """For the demo: None when roi_pool.py or proposals.py is not in the project yet; "roi_pool" when the learner's
    roi_pool stops with an error or does not give (64, K, K) numbers that training can follow back for a box of the
    scene; "proposals" when their proposals stops or does not give boxes; "offsets" when their boxes_from stops on the
    trained detector's offsets; "slow" when the training takes too long; else a dict with the loss of each stage after
    every epoch ("curves"), the boxes on the SHOWN scenes after every epoch ("moments"), the summary of the trained
    detector on the test scenes, its parameters, and the modules the course's versions stood in for. `show` hears the
    run while it trains."""
    if not (_saved("roi_pool") and _saved("proposals")):
        return None
    names = {}
    for module, name, key in (("roi_pool", "roi_pool", "roi_pool"), ("proposals", "proposals", "proposals"),
                              ("offsets", "boxes_from", "offsets")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    ss, X = scene_set("train")
    torch.manual_seed(0)
    model = TwoStage()
    try:   # one scene through both: the shapes the training needs
        maps, rows = model(X[:1].float() / 255)
        props = as_boxes(names["proposals"](rows[0].detach(), anchors().clone(), TRAIN_KEEP))
        if not (1 <= len(props) <= TRAIN_KEEP and torch.isfinite(props).all()):
            return "proposals"
    except Exception:
        return "proposals"
    try:
        box = tuple(boxes_and_labels(ss[0])[0][0].tolist())
        p = names["roi_pool"](maps[0], box, K)
        if not (torch.is_tensor(p) and tuple(p.shape) == (64, K, K) and p.requires_grad and torch.isfinite(p).all()):
            return "roi_pool"
    except Exception:
        return "roi_pool"
    pool, props_of, boxes_of = (_named(names["roi_pool"], "roi_pool"), _named(names["proposals"], "proposals"),
                                _named(names["boxes_from"], "offsets"))
    torch.manual_seed(0)   # the same start as the course's run
    model = TwoStage()
    moments, broken = [], []

    def heard(p):
        if "model" in p:
            try:
                moments.append(shown_moment(p["model"], pool, props_of, boxes_of))
            except _Stopped as e:
                broken.append(e.key)
                moments.append([[] for _ in SHOWN])
            except Exception:
                broken.append("offsets")
                moments.append([[] for _ in SHOWN])
            p = {k: v for k, v in p.items() if k != "model"}
        if show is not None:
            try:
                show(p)
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass

    try:   # with_time_limit trains on one thread, as the course's run
        history = with_time_limit(lambda: train(model, 0, EPOCHS, heard, pool, props_of), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    except Exception:   # the pooled numbers of a box did not fit the second stage
        return "roi_pool"
    if broken:
        return broken[0]
    test_ss, Xt = scene_set("test")
    try:
        result = summary(detect(model, Xt, TEST_KEEP, pool, props_of, boxes_of), test_ss)
    except _Stopped as e:
        return e.key
    except Exception:
        return "offsets"
    curves = {"rpn": [float(h["rpn_loss"]) for h in history], "second": [float(h["second_loss"]) for h in history]}
    return result | {"curves": curves, "moments": moments, "params": sum(v.numel() for v in model.parameters()), "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("roi_pool", "proposals", "offsets"):
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
    """Everything TRAINED holds, measured again: every seed with RoI pooling, and seed 0 with roi_align. Takes about
    three minutes per seed on one thread, and four with roi_align."""
    runs = {seed: measure_seed(seed) for seed in seeds}
    align = measure_seed(0, roi_align)
    keys = ("map", "aps", "map75", "found", "false", "covered")
    out = {key: [runs[seed][key] for seed in seeds] for key in keys}
    first = runs[min(seeds)]
    out |= {"shapes": first["shapes"], "curves": first["curves"], "scene": first["scene"],
            "keeps": {n: {key: v for key, v in stop.items() if key != "shapes"} for n, stop in first["keeps"].items()},
            "align": {"map": align["map"], "map75": align["map75"], "found": align["found"], "false": align["false"],
                      "keeps": {n: {key: stop[key] for key in ("map", "map75")} for n, stop in align["keeps"].items()}}}
    return out
