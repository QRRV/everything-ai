# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-one-stage-detector, a step of the project cnn: targets, which turns the true boxes of a
scene into what every anchor learns (a class and a box offset), detection_loss, which adds a loss for the kinds and
one for the boxes, and Detector, a CNN that gives every anchor of a scene a kind and a box offset at once. Each is
saved in the learner's project (work/projects/cnn/targets.py, detection_loss.py and detector.py) when right. The demo
trains the learner's Detector with their targets and loss on the course's scenes, finds the shapes of new scenes
with their boxes_from and NMS, and measures the mAP. And the pictures: the boxes of the course's trained detector on
the scene of the worked example before and after NMS, and for the slider, what the weight of the box loss does.
Everything is computed from the course's own scenes (scenes.py): no download."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch import nn

from everything_ai import chart, projects, scenes
from everything_ai.check import ClassBox, Mistake, ScenarioBox, Timeout, run_code, with_time_limit
from everything_ai import netview, yours

# The demo uses the learner's match and offsets of cv-anchors: targets imports them. Where a file is missing or
# broken, the course's version stands in, and the lesson says so.
BUILDS_ON = (("cnn", "match", "cv-anchors", "match"), ("cnn", "offsets", "cv-anchors", "boxes-from"))
projects.builds_on(*BUILDS_ON)

SEED = 18                     # the scene of the worked example, as in cv-annotations, cv-iou, cv-nms and cv-anchors
CELLS, STRIDE = 16, 8         # a feature map of 16 by 16 squares, each for 8 by 8 pixels of the scene of 128 by 128
SIZES = ((16, 16), (24, 24), (32, 32))   # the anchors of every square, as in cv-anchors: 768 anchors
KINDS = len(scenes.KINDS)     # circle, square, triangle: classes 1, 2 and 3; class 0 is background
PER = 1 + KINDS + 4           # numbers per anchor: a logit for background and for each kind, and the 4 offsets
CIRCLE_ANCHOR = 458           # the circle's best anchor in the worked example, on square (9, 8), of 32 pixels
N_TRAIN, N_TEST = 2000, 200   # training scenes 1000 to 2999, test scenes 0 to 199 (scene 18 among them)
TRAIN, TEST = range(1000, 1000 + N_TRAIN), range(N_TEST)
EPOCHS, LR, BATCH = 8, 0.01, 32
TARGETS_LIMIT, TRAIN_LIMIT = 120, 600   # seconds the demo gives the learner's targets and the training: the notes say two and ten minutes
SCORE, NMS_IOU = 0.5, 0.5     # the score a box needs to be shown, and NMS's threshold, as in cv-nms
LOW = 0.05                    # the lowest score that counts for the mAP, as torchvision's detectors keep by default
AP_IOU = 0.5                  # the IoU above which a box is a true positive, as in cv-map
SHOWN = (18, 5, 11)           # the test scenes of the rewind
SHOWN_SHAPES = sum(len(scenes.scene(seed).shapes) for seed in SHOWN)
WEIGHTS = (0.0, 0.25, 1.0, 4.0, 16.0)   # the weights of the box loss on the slider; 1 is the lesson's loss

# The numbers of the worked example, measured without training. `measure()` measures them again; the test of the
# lesson checks that both are the same.
REFERENCE = {
    "classes": (735, 20, 13),           # anchors of the scene that learn background, a shape, that are ignored
    "circle": (1, (-0.1094, -0.1094, -0.0984, -0.0984)),   # the class and the offsets of the circle's best anchor
    "kinds": (2, 2, 1, 0),              # the label of each shape of the scene: two triangles, a square, a circle
    "shapes": ((1, 3, 128, 128), (1, 24, 16, 16), (1, 768, 8)),   # the scene, the head's maps, the 768 rows
    "row": (0.3509, -0.7183, -0.5193, -0.7151, -0.9078, -2.1249, 0.4139, -1.0114),   # the untrained row of anchor 458
    "plain": (0.1454, 0.2703, 0.4176, -0.1661, 0.0089, 0.4904, -0.4533, 0.7628),   # the same with reshape alone
    "losses": (755, 1.4667, 0.6996),    # anchors not ignored, the kind loss and the box loss, untrained
    "untrained": ((0.4747, 0.163, 0.1988, 0.1635), 0.573, 16),   # anchor 458's probabilities, the top score, >= 0.5
    "params": 38616,
}

# What the course measured, on one thread, with the course's code: Detector after torch.manual_seed(seed), the
# training scenes shuffled with a generator seeded with seed + 1, in mini-batches of BATCH, AdamW with a learning rate
# of LR, a warmup of one epoch and cosine decay, as the Trainer of nn-trainer, and gradients clipped at 1.0, for the
# seeds 0 to 4. "map" is the mAP at an IoU of 0.5 over the 200 test scenes, with every box down to a score of LOW;
# "aps" the AP of each kind; "map75" the mAP at 0.75; "found" how many of the shapes of the test scenes get a box of
# their kind with a score of 0.5 or more and an IoU above 0.5, "false" how many boxes of 0.5 or more find no shape;
# "before" the boxes of 0.5 or more per scene before NMS; "val" the validation loss after the last epoch. "curves" is
# seed 0, the loss after every epoch. "scene" is seed 0 on the scene of the worked example: before NMS every anchor
# with a score of 0.5 or more (anchor, kind, score, box), after NMS the boxes that stay. "weights" is seed 0 with
# another weight on the box loss: its mAP at 0.5 and at 0.75, and its boxes of 0.5 or more on the scene after NMS.
# tests/test_cv_one_stage_detector.py measures seed 0 again; measure_trained() measures all of it.
TRAINED = {'map': [0.9159, 0.9119, 0.9137, 0.9182, 0.9059],
           'aps': [[0.9426, 0.892, 0.9131],
                   [0.9338, 0.8832, 0.9187],
                   [0.9391, 0.8674, 0.9347],
                   [0.9481, 0.8942, 0.9122],
                   [0.9418, 0.8692, 0.9068]],
           'map75': [0.8192, 0.8068, 0.8149, 0.822, 0.8076],
           'found': [733, 726, 725, 749, 726],
           'false': [120, 112, 130, 116, 116],
           'before': [20.45, 20.07, 19.77, 20.46, 19.64],
           'val': [0.051, 0.0514, 0.0522, 0.0511, 0.0521],
           'shapes': 808,
           'curves': {'train': [0.3323, 0.1003, 0.0844, 0.0714, 0.062, 0.056, 0.0511, 0.0491],
                      'val': [0.1228, 0.1076, 0.0825, 0.0697, 0.0615, 0.0575, 0.0518, 0.051]},
           'scene': {'before': [(307, 1, 0.743, (41.4, 39.8, 67.7, 67.7)),
                                (308, 1, 0.73, (40.4, 38.8, 69.1, 68.4)),
                                (310, 2, 0.567, (42.2, 43.3, 64.6, 70.6)),
                                (311, 2, 0.75, (41.6, 43.0, 67.5, 71.6)),
                                (343, 2, 0.939, (51.0, 7.9, 73.9, 29.3)),
                                (344, 2, 0.934, (50.6, 7.5, 73.9, 29.4)),
                                (356, 1, 0.539, (41.8, 36.9, 73.4, 68.6)),
                                (390, 2, 0.554, (53.0, 7.5, 76.5, 27.9)),
                                (391, 2, 0.913, (51.5, 8.5, 75.1, 29.3)),
                                (392, 2, 0.854, (51.5, 7.5, 75.4, 29.2)),
                                (407, 0, 0.941, (53.5, 47.1, 85.8, 78.8)),
                                (409, 0, 0.7, (56.0, 49.8, 86.4, 78.8)),
                                (410, 0, 0.952, (55.2, 48.7, 86.8, 79.5)),
                                (454, 0, 0.9, (59.3, 50.2, 87.3, 79.3)),
                                (455, 0, 0.982, (57.2, 49.3, 86.8, 79.8)),
                                (457, 0, 0.947, (57.5, 50.1, 86.2, 78.9)),
                                (458, 0, 0.987, (56.8, 49.7, 86.2, 78.2))],
                     'after': [(0, 0.987, (56.8, 49.7, 86.2, 78.2)),
                               (2, 0.939, (51.0, 7.9, 73.9, 29.3)),
                               (2, 0.75, (41.6, 43.0, 67.5, 71.6)),
                               (1, 0.743, (41.4, 39.8, 67.7, 67.7))]},
           'weights': {0.0: {'map': 0.0053,
                             'map75': 0.0,
                             'boxes': [(0, 0.999, (4.4, 60.5, 49.0, 86.6)),
                                       (0, 0.997, (83.3, 66.9, 125.4, 87.9)),
                                       (2, 0.994, (81.3, 22.0, 96.8, 42.0)),
                                       (0, 0.99, (36.9, 55.4, 56.8, 68.7)),
                                       (2, 0.988, (56.5, -21.4, 76.7, -5.7)),
                                       (0, 0.985, (23.9, 42.0, 61.7, 53.8)),
                                       (2, 0.981, (65.3, -1.3, 84.4, 45.8)),
                                       (2, 0.98, (43.6, -20.9, 71.2, -6.6)),
                                       (0, 0.969, (56.0, 8.1, 73.2, 28.7)),
                                       (0, 0.966, (41.0, 27.5, 120.9, 58.3)),
                                       (0, 0.935, (93.0, 58.8, 107.5, 104.6)),
                                       (1, 0.874, (68.2, 52.6, 86.6, 72.3)),
                                       (1, 0.872, (22.1, 19.1, 63.7, 38.9)),
                                       (1, 0.793, (48.2, 39.5, 74.5, 75.6)),
                                       (1, 0.77, (26.4, 41.5, 53.9, 80.3)),
                                       (1, 0.766, (44.9, 28.5, 61.5, 53.3)),
                                       (1, 0.698, (8.5, 23.0, 52.5, 42.6)),
                                       (2, 0.631, (104.7, -5.5, 119.8, 22.8)),
                                       (2, 0.615, (101.5, 9.8, 140.3, 40.2)),
                                       (0, 0.563, (46.7, 32.5, 86.4, 63.2)),
                                       (2, 0.527, (107.9, 15.3, 128.9, 31.2))]},
                       0.25: {'map': 0.9473,
                              'map75': 0.8613,
                              'boxes': [(2, 0.989, (51.0, 7.0, 76.0, 29.6)),
                                        (0, 0.987, (56.7, 48.4, 86.5, 77.6)),
                                        (1, 0.953, (42.0, 39.6, 68.0, 65.9)),
                                        (2, 0.565, (40.3, 42.4, 67.3, 71.3)),
                                        (2, 0.553, (102.8, 6.7, 115.4, 22.1))]},
                       1.0: {'map': 0.9159,
                             'map75': 0.8192,
                             'boxes': [(0, 0.987, (56.8, 49.7, 86.2, 78.2)),
                                       (2, 0.939, (51.0, 7.9, 73.9, 29.3)),
                                       (2, 0.75, (41.6, 43.0, 67.5, 71.6)),
                                       (1, 0.743, (41.4, 39.8, 67.7, 67.7))]},
                       4.0: {'map': 0.7898,
                             'map75': 0.7118,
                             'boxes': [(2, 0.863, (51.0, 7.5, 75.9, 30.6)),
                                       (1, 0.6, (57.3, 48.9, 85.9, 78.1)),
                                       (1, 0.583, (43.2, 43.0, 75.1, 75.1)),
                                       (0, 0.511, (54.7, 47.1, 85.1, 78.3))]},
                       16.0: {'map': 0.6748, 'map75': 0.5979, 'boxes': [(2, 0.696, (51.2, 6.6, 75.1, 29.2))]}},
           'misses': {'small': (25, 95), 'large': (32, 550), 'wrong_kind': 96, 'after': 4.26}}


def low_high(values) -> tuple:
    return min(values), max(values)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def whole(n: int, lang: str) -> str:
    """A whole number with a mark between the thousands: 38,616 in English, 38.616 in Dutch."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def percent(part: float, total: float, digits: int = 0) -> str:
    return f"{100 * part / total:.{digits}f}"


# --- the course's versions ------------------------------------------------------------------------------------------

def iou_matrix(anchors, boxes) -> np.ndarray:
    """The IoU of every anchor with every box, as in cv-anchors: shape (anchors, boxes)."""
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
    out = np.zeros((len(anchors), len(boxes)))
    area_a = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    for k in range(len(boxes)):
        box = boxes[k]
        top, left = np.maximum(anchors[:, 0], box[0]), np.maximum(anchors[:, 1], box[1])
        bottom, right = np.minimum(anchors[:, 2], box[2]), np.minimum(anchors[:, 3], box[3])
        overlap = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)
        out[:, k] = overlap / (area_a + (box[2] - box[0] + 1) * (box[3] - box[1] + 1) - overlap)
    return out


def match(ious) -> np.ndarray:
    """What each anchor learns, as in cv-anchors: a box, -1 for background, -2 when it is ignored."""
    ious = np.asarray(ious, dtype=float)
    best, most = ious.argmax(axis=1), ious.max(axis=1)
    label = np.zeros(len(ious), dtype=int) - 1
    label[most >= 0.5] = best[most >= 0.5]
    label[(most >= 0.4) & (most < 0.5)] = -2
    for k in range(ious.shape[1]):
        label[ious[:, k] == ious[:, k].max()] = k
    return label


def offsets(anchors, boxes):
    """The box offsets of boxes from their anchors, as in cv-anchors, in PyTorch."""
    middle_a, size_a = (anchors[:, :2] + anchors[:, 2:]) / 2, anchors[:, 2:] - anchors[:, :2] + 1
    middle, size = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2] + 1
    return torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)


def boxes_from(anchors, t):
    """The boxes that the offsets t make of the anchors, as in cv-anchors."""
    middle_a, size_a = (anchors[:, :2] + anchors[:, 2:]) / 2, anchors[:, 2:] - anchors[:, :2] + 1
    middle, size = middle_a + t[:, :2] * size_a, size_a * torch.exp(t[:, 2:])
    return torch.cat([middle - (size - 1) / 2, middle + (size - 1) / 2], dim=1)


def targets(anchors, boxes, labels):
    """What every anchor learns: its class, 0 for background, the kind + 1 for an anchor that learns a box, -1 when it
    is ignored; and its box offset, for the anchors that learn a box (0 for the others)."""
    label = torch.as_tensor(match(iou_matrix(anchors.numpy(), boxes.numpy())))
    learn = label >= 0
    classes = torch.zeros(len(anchors), dtype=torch.int64)
    classes[learn] = labels[label[learn]] + 1
    classes[label == -2] = -1
    t = torch.zeros(len(anchors), 4)
    t[learn] = offsets(anchors[learn], boxes[label[learn]])
    return classes, t


def detection_loss(out, classes, t, weight: float = 1.0):
    """The cross-entropy of the logits over the anchors that are not ignored, plus the mean absolute difference of the
    offsets over the anchors that learn a box, that one times `weight` (1 in the lesson; the slider tries others)."""
    kept = classes >= 0
    kind_loss = F.cross_entropy(out[:, :, :4][kept], classes[kept])
    learn = classes > 0
    box_loss = (out[:, :, 4:][learn] - t[learn]).abs().mean()
    return kind_loss + weight * box_loss


class Detector(nn.Module):
    """The course's one-stage detector: three poolings make 64 feature maps of 16 by 16, and a convolution of 3 by 3
    gives every square 8 numbers for each of its 3 anchors."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 3 * 8, 3, padding=1)

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out.permute(0, 2, 3, 1).reshape(len(x), -1, 8)


# --- scenes and their targets ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def anchors() -> torch.Tensor:
    return torch.tensor(scenes.anchor_boxes(CELLS, STRIDE, SIZES), dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    return scenes.scene(SEED)


def boxes_and_labels(s: scenes.Scene) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor([shape.box for shape in s.shapes], dtype=torch.float32),
            torch.tensor([shape.label for shape in s.shapes]))


@functools.lru_cache(maxsize=2)
def scene_set(which: str) -> tuple:
    """The training or the test scenes, and their pictures as one tensor of bytes, shape (N, 3, 128, 128)."""
    ss = tuple(scenes.scene(seed) for seed in (TRAIN if which == "train" else TEST))
    return ss, torch.tensor(np.stack([s.image for s in ss])).permute(0, 3, 1, 2).contiguous()


def all_targets(ss, targets_fn=targets) -> tuple[torch.Tensor, torch.Tensor]:
    """The classes (N, 768) and the offsets (N, 768, 4) of scenes, from a targets function."""
    a = anchors()
    classes, ts = [], []
    for s in ss:
        c, t = targets_fn(a.clone(), *boxes_and_labels(s))
        classes.append(torch.as_tensor(c).long().reshape(len(a)))
        ts.append(torch.as_tensor(t, dtype=torch.float32).reshape(len(a), 4))
    return torch.stack(classes), torch.stack(ts)


@functools.lru_cache(maxsize=1)
def course_targets() -> tuple:
    return all_targets(scene_set("train")[0]), all_targets(scene_set("test")[0])


# --- training, detecting, measuring ---------------------------------------------------------------------------------

def lr_factor(step: int, warmup: int, total: int) -> float:
    """Warmup and cosine decay, as the Trainer of nn-trainer."""
    if step < warmup:
        return (step + 1) / warmup
    if step >= total:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def outputs(model, X) -> torch.Tensor:
    """The model's 768 rows for every scene of X (bytes), in eval mode, without a graph."""
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(X[i:i + 50].float() / 255) for i in range(0, len(X), 50)])
    model.train()
    return out


def train(model, loss_fn, data, test, seed: int = 0, epochs: int = EPOCHS, heard=None) -> list[dict]:
    """Train `model` on data = (X, classes, t), as the course measured, and measure the validation loss on test after
    every epoch. `heard` hears every 10 batches how far an epoch is, and after every epoch its validation loss with the
    outputs on the test scenes."""
    X, C, T = data
    Xt, Ct, Tt = test
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
        running = 0.0
        for b, i in enumerate(range(0, n, BATCH), 1):
            idx = order[i:i + BATCH]
            loss = loss_fn(model(X[idx].float() / 255), C[idx], T[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if heard is not None and b % 10 == 0:
                heard({"epoch": epoch, "epochs": epochs, "batch": b, "batches": per})
        out = outputs(model, Xt)
        val = float(loss_fn(out, Ct, Tt))
        history.append({"epoch": epoch, "train_loss": running / per, "val_loss": val})
        if heard is not None:
            heard({"epoch": epoch, "epochs": epochs, "val_loss": val, "out": out})
    return history


def iou(a, b) -> float:
    """The intersection over union of two boxes, as in cv-iou."""
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    both = max(0, bottom - top + 1) * max(0, right - left + 1)
    return both / ((a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - both)


def nms(boxes, scores, threshold: float) -> list[int]:
    """Non-maximum suppression, as in cv-nms: the numbers of the boxes kept, the highest score first."""
    rest, kept = list(range(len(boxes))), []
    while rest:
        best = rest[0]
        for i in rest:
            if scores[i] > scores[best]:
                best = i
        kept.append(best)
        rest = [i for i in rest if iou(boxes[i], boxes[best]) <= threshold]
    return kept


def scored(out) -> tuple[torch.Tensor, torch.Tensor]:
    """For every anchor of one scene (out of shape (768, 8)): its score, the highest probability of a kind, and that
    kind (0, 1 or 2), from the softmax over its 4 logits."""
    p = out[:, :4].softmax(dim=1)
    score, kind = p[:, 1:].max(dim=1)
    return score, kind


def detect(out, score: float = LOW, threshold: float = NMS_IOU, boxes_fn=boxes_from) -> list[tuple]:
    """The boxes of one scene, (kind, score, box), the highest score first: every anchor with a score of `score` or
    more, turned into its box with boxes_fn, then NMS for each kind on its own."""
    s, k = scored(out)
    keep = s >= score
    boxes = torch.as_tensor(boxes_fn(anchors()[keep], out[keep, 4:].clone()), dtype=torch.float32).reshape(-1, 4).tolist()
    s, k = s[keep].tolist(), k[keep].tolist()
    found = []
    for kind in range(KINDS):
        mine = [i for i in range(len(boxes)) if k[i] == kind]
        for j in nms([boxes[i] for i in mine], [s[i] for i in mine], threshold):
            found.append((kind, s[mine[j]], tuple(boxes[mine[j]])))
    found.sort(key=lambda d: -d[1])
    return found


def hits(found: list, s: scenes.Scene, threshold: float = AP_IOU) -> list[bool]:
    """For each box of one scene, the highest score first, whether it finds a shape: its IoU with a shape of its kind
    is above the threshold, and no box with a higher score found that shape first (cv-map)."""
    taken, out = set(), []
    for kind, _, box in found:
        shape = None
        for q, sh in enumerate(s.shapes):
            if sh.label == kind and iou(box, sh.box) > threshold:
                shape = q
        out.append(shape is not None and shape not in taken)
        if shape is not None:
            taken.add(shape)
    return out


def mean_ap(found_per_scene: list, ss, threshold: float = AP_IOU) -> tuple[float, list[float]]:
    """The mAP over scenes, as cv-map computes it: for each kind, the precision at every true positive, added up and
    divided by the number of shapes of that kind; then the mean over the kinds."""
    aps = []
    for kind in range(KINDS):
        boxes = sorted(((img, sc, b) for img, f in enumerate(found_per_scene) for k, sc, b in f if k == kind), key=lambda d: -d[1])
        n = sum(1 for s in ss for sh in s.shapes if sh.label == kind)
        taken, tp, total = set(), 0, 0.0
        for r, (img, _, b) in enumerate(boxes):
            shape = None
            for q, sh in enumerate(ss[img].shapes):
                if sh.label == kind and iou(b, sh.box) > threshold:
                    shape = (img, q)
            if shape is not None and shape not in taken:
                taken.add(shape)
                tp += 1
                total += tp / (r + 1)
        aps.append(total / n)
    return float(np.mean(aps)), aps


def summary(out, ss, boxes_fn=boxes_from) -> dict:
    """What the lesson says of a trained detector, from its outputs on the test scenes."""
    found = [detect(out[i], boxes_fn=boxes_fn) for i in range(len(ss))]
    m, aps = mean_ap(found, ss)
    m75, _ = mean_ap(found, ss, 0.75)
    right = false = 0
    before = []
    for i, s in enumerate(ss):
        shown = [d for d in found[i] if d[1] >= SCORE]
        h = hits(shown, s)
        right += sum(h)
        false += len(h) - sum(h)
        before.append(int((scored(out[i])[0] >= SCORE).sum()))
    return {"map": round(m, 4), "aps": [round(a, 4) for a in aps], "map75": round(m75, 4), "found": right, "false": false,
            "shapes": sum(len(s.shapes) for s in ss), "before": round(float(np.mean(before)), 2)}


def misses(out, ss) -> dict:
    """What a trained detector misses, and where its false boxes lie, from its outputs on the test scenes: of the shapes
    whose box is at most 20 pixels on its longest side, and of those of 25 pixels or more, how many it misses and how
    many there are; how many of its false boxes of 0.5 or more lie on a shape (IoU above 0.5) with another kind; and
    its boxes of 0.5 or more per scene after NMS."""
    small = large = small_missed = large_missed = wrong_kind = boxes = 0
    for i, s in enumerate(ss):
        shown = [d for d in detect(out[i]) if d[1] >= SCORE]
        boxes += len(shown)
        found = set()
        for (kind, _, box), hit in zip(shown, hits(shown, s)):
            if hit:   # the shape hits() gave it: the last one of its kind above the threshold
                found.add([q for q, sh in enumerate(s.shapes) if sh.label == kind and iou(box, sh.box) > AP_IOU][-1])
            elif any(sh.label != kind and iou(box, sh.box) > AP_IOU for sh in s.shapes):
                wrong_kind += 1
        for q, sh in enumerate(s.shapes):
            longest = max(sh.box[2] - sh.box[0] + 1, sh.box[3] - sh.box[1] + 1)
            if longest <= 20:
                small, small_missed = small + 1, small_missed + (q not in found)
            elif longest >= 25:
                large, large_missed = large + 1, large_missed + (q not in found)
    return {"small": (small_missed, small), "large": (large_missed, large), "wrong_kind": wrong_kind,
            "after": round(boxes / len(ss), 2)}


def shown_boxes(out, boxes_fn=boxes_from) -> list[tuple]:
    """The boxes of one scene with a score of SCORE or more, after NMS, rounded for the pictures."""
    return [(k, round(s, 3), tuple(round(v, 1) for v in b)) for k, s, b in detect(out, boxes_fn=boxes_fn) if s >= SCORE]


def train_course(seed: int = 0, weight: float = 1.0, heard=None) -> tuple:
    """The course's detector, trained as the course measured: on one thread, as a check runs."""
    (C, T), (Ct, Tt) = course_targets()
    _, X = scene_set("train")
    _, Xt = scene_set("test")
    torch.manual_seed(seed)
    model = Detector()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        history = train(model, lambda o, c, t: detection_loss(o, c, t, weight), (X, C, T), (Xt, Ct, Tt), seed, EPOCHS, heard)
    finally:
        torch.set_num_threads(threads)
    return model, history


def measure_trained(seeds=range(5)) -> dict:
    """Every number of TRAINED, measured again: 5 + len(WEIGHTS) - 1 trainings, several minutes each on one thread."""
    ss, Xt = scene_set("test")
    out: dict = {"map": [], "aps": [], "map75": [], "found": [], "false": [], "before": [], "val": []}
    for seed in seeds:
        model, history = train_course(seed)
        o = outputs(model, Xt)
        s = summary(o, ss)
        for key in ("map", "aps", "map75", "found", "false", "before"):
            out[key].append(s[key])
        out["val"].append(round(history[-1]["val_loss"], 4))
        if seed == 0:
            out["shapes"] = s["shapes"]
            out["curves"] = {"train": [round(h["train_loss"], 4) for h in history], "val": [round(h["val_loss"], 4) for h in history]}
            row = o[SEED]
            score, kind = scored(row)
            boxes = boxes_from(anchors(), row[:, 4:])
            out["scene"] = {"before": [(i, int(kind[i]), round(float(score[i]), 3), tuple(round(float(v), 1) for v in boxes[i]))
                                       for i in range(len(row)) if score[i] >= SCORE],
                            "after": shown_boxes(row)}
            out["weights"] = {1.0: {"map": s["map"], "map75": s["map75"], "boxes": out["scene"]["after"]}}
            out["misses"] = misses(o, ss)
    for w in WEIGHTS:
        if w == 1.0:
            continue
        model, _ = train_course(0, w)
        o = outputs(model, Xt)
        s = summary(o, ss)
        out["weights"][w] = {"map": s["map"], "map75": s["map75"], "boxes": shown_boxes(o[SEED])}
    return out


def measure() -> dict:
    """Every number of REFERENCE, measured again: the worked example, without training."""
    s = example()
    a = anchors()
    truth, kinds = boxes_and_labels(s)
    classes, t = targets(a, truth, kinds)
    torch.manual_seed(0)
    net = Detector()
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        maps = net.head(net.backbone(x))
        out = maps.permute(0, 2, 3, 1).reshape(1, 768, 8)
        plain = maps.reshape(1, 768, 8)
        kept = classes >= 0
        learn = classes > 0
        kind_loss = F.cross_entropy(out[0, kept, :4], classes[kept])
        box_loss = (out[0, learn, 4:] - t[learn]).abs().mean()
        e = torch.exp(out[0, :, :4])   # softmax, as the worked example writes it
        p = e / e.sum(dim=1, keepdim=True)
        score = p[:, 1:].max(dim=1).values
    r4 = lambda v: tuple(round(float(x), 4) for x in v)   # noqa: E731
    return {
        "classes": (int((classes == 0).sum()), int((classes > 0).sum()), int((classes == -1).sum())),
        "circle": (int(classes[CIRCLE_ANCHOR]), r4(t[CIRCLE_ANCHOR])),
        "kinds": tuple(int(k) for k in kinds),
        "shapes": (tuple(x.shape), tuple(maps.shape), tuple(out.shape)),
        "row": r4(out[0, CIRCLE_ANCHOR]),
        "plain": r4(plain[0, CIRCLE_ANCHOR]),
        "losses": (int(kept.sum()), round(float(kind_loss), 4), round(float(box_loss), 4)),
        "untrained": (r4(p[CIRCLE_ANCHOR]), round(float(score.max()), 4), int((score >= SCORE).sum())),
        "params": sum(v.numel() for v in net.parameters()),
    }


# --- pictures ---------------------------------------------------------------------------------------------------------

CROP = (32, 0, 127, 95)       # the part of the scene of the worked example with its shapes, as in cv-anchors


def _photo_axes(ax, img, title, crop=CROP, size=9):
    ax.imshow(img, interpolation="nearest")
    top, left, bottom, right = crop
    ax.set_xlim(left - 0.5, right + 0.5)
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("none")
    ax.set_title(title, fontsize=size)


def _draw_box(ax, b, colour, linestyle, width=1.5, halo=3.2):
    """A box as a line around its pixels, with a thin edge in the color of the page, so that it stands out on every
    shape, in light and in dark."""
    from everything_ai import plots
    top, left, bottom, right = b
    xs = [left - 0.5, right + 0.5, right + 0.5, left - 0.5, left - 0.5]
    ys = [top - 0.5, top - 0.5, bottom + 0.5, bottom + 0.5, top - 0.5]
    lines = ax.plot(xs, ys, color=plots.PAGE, linestyle="-", linewidth=halo, solid_joinstyle="miter")
    lines += ax.plot(xs, ys, color=colour, linestyle=linestyle, linewidth=width, solid_joinstyle="miter", dash_capstyle="butt")
    for line in lines:   # a box may stick out of the picture: it is cut off there, and the picture keeps its size
        line.set_in_layout(False)


def _truth_box(ax, b):
    """A true box: solid, in the color of data."""
    from everything_ai import plots
    _draw_box(ax, b, plots.DATA, "-", width=1.3, halo=2.8)


def _found_box(ax, b, width=1.5, halo=3.2):
    """A box the detector predicts: dashed, in the color of a prediction."""
    from everything_ai import plots
    _draw_box(ax, b, plots.PRED, (0, (3, 1.6)), width=width, halo=halo)


def _score(ax, b, text, size=9):
    """A box's words, just above its top left corner, on a band in the color of the page."""
    from everything_ai import plots
    t = ax.text(b[1] - 0.5, b[0] - 1.2, text, fontsize=size, color=plots.PRED_TEXT, va="bottom", ha="left",
                bbox={"boxstyle": "square,pad=0.1", "facecolor": plots.PAGE, "edgecolor": "none", "alpha": 0.85})
    t.set_in_layout(False)
    return t


def _clear_of(ax, t, b, placed, renderer):
    """Move the words of box `b` off the words of the boxes before it: to their right, else above them, else under
    the box, so that two boxes with almost the same top left corner each keep their words readable. Words stay inside
    the picture."""
    room = ax.get_window_extent(renderer)
    to_data = ax.transData.inverted()
    under = False
    for _ in range(len(placed) + 1):
        mine = t.get_window_extent(renderer)
        hit = [box for box in (o.get_window_extent(renderer) for o in placed) if box.overlaps(mine)]
        if not hit:
            return
        right = max(box.x1 for box in hit) + 6   # in pixels of the figure
        up = max(box.y1 for box in hit) + 3      # upward
        if right + mine.width <= room.x1:
            t.set_x(to_data.transform((right, 0))[0])
        elif up + mine.height <= room.y1 and not under:
            t.set_y(to_data.transform((0, up))[1])
        elif not under:
            t.set_position((b[1] - 0.5, b[2] + 1.2))
            t.set_va("top")
            under = True
        else:
            return


def _side_by_side(panels: list[tuple[str, int]], alt: str) -> str:
    """Pictures next to each other, that go below each other on a phone, so that their text keeps its size: each
    rendered on its own, with the whole description on the row."""
    import html
    parts = "".join(f'<div style="flex:1 1 {basis}px;min-width:0">{svg}</div>' for svg, basis in panels)
    return (f'<div role="img" aria-label="{html.escape(alt, quote=True)}" style="display:flex;flex-wrap:wrap;gap:4px 20px;'
            f'align-items:flex-start">{parts}</div>')


def _scene_svg(s, boxes, title, names=None, figsize=(3.9, 4.05), crop=CROP, size=10, labels=True):
    """One scene with its true boxes and the detector's boxes (kind, score, box); with `names`, each box with its kind
    and score."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    fig, ax = plt.subplots(figsize=figsize)
    _photo_axes(ax, s.image, title, crop=crop, size=size)
    for shape in s.shapes:
        _truth_box(ax, shape.box)
    for _, _, b in boxes:
        _found_box(ax, b)
    if names is not None and labels:
        placed, renderer = [], fig.canvas.get_renderer()
        for k, sc, b in boxes:
            words = _score(ax, b, names["box"].format(kind=names["kinds"][k], score=f"{sc:.2f}".replace(".", names["point"])))
            _clear_of(ax, words, b, placed, renderer)
            placed.append(words)
    svg = plots.render(fig)
    plt.close(fig)
    return svg


def nms_picture(words: dict[str, str]) -> str:
    """Two pictures of the scene of the worked example and the course's trained detector (seed 0): every anchor's box
    with a score of 0.5 or more, before NMS; and the boxes that NMS keeps, with their kind and score."""
    s = example()
    before = [(k, sc, b) for _, k, sc, b in TRAINED["scene"]["before"]]
    after = TRAINED["scene"]["after"]
    left = _scene_svg(s, before, words["before"].format(n=len(before)))
    right = _scene_svg(s, after, words["after"].format(n=len(after)), names=words)
    return _side_by_side([(left, 300), (right, 300)], words["alt"])


def weight_frame(w: float, words: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the boxes of 0.5 or more after NMS on the scene of the worked example, of the course's detector
    trained with weight `w` on the box loss; next to it its mAP at an IoU of 0.5 and of 0.75 at every weight."""
    stop = TRAINED["weights"][w]
    s = example()
    text = weight_text(w, words["point"])
    scene = _scene_svg(s, stop["boxes"], words["title"].format(weight=text, n=len(stop["boxes"])),
                       names=words, figsize=(2.9, 3.0), size=9, labels=len(stop["boxes"]) <= 8)   # 21 at 0: boxes only
    xs = list(range(len(WEIGHTS)))   # the weights at equal steps: each is 4 times the one before, after 0
    p = chart.Panel((-0.4, len(WEIGHTS) - 0.6), (0, 1.05), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], title=words["chart"], xlabel=words["weight"], ylabel="mAP",
                    xticks=xs, xtick_labels=[weight_text(v, words["point"]) for v in WEIGHTS])
    p.line(xs, [TRAINED["weights"][v]["map"] for v in WEIGHTS], "ink", 2.2)
    p.line(xs, [TRAINED["weights"][v]["map75"] for v in WEIGHTS], "ink", 2.2, dashed=True)
    i = WEIGHTS.index(w)
    p.points([i, i], [stop["map"], stop["map75"]], "ink", r=4.5)
    p.text(2, TRAINED["weights"][1.0]["map"] + 0.08, words["at50"], "ink")    # above the solid line, at weight 1
    p.text(2, TRAINED["weights"][1.0]["map75"] - 0.1, words["at75"], "ink")   # below the dashed line
    plot = chart.figure([p], alt=words["chart_alt"], width=276, height=236)   # 11 px text in a play box on a phone
    if words["point"] == ",":
        plot = chart.with_decimal_comma(plot)
    return _side_by_side([(scene, 250), (plot, 276)], words["alt"].format(weight=text, n=len(stop["boxes"]))), stop


def weight_text(w: float, point: str) -> str:
    """A weight of the slider as the lesson writes it: 0, 0.25, 1, 4 or 16."""
    return (f"{w:g}").replace(".", point)


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the training scenes solid and on the test scenes dashed, as in
    torch-loss-curves. With `at`, a line at that epoch, for rewinding."""
    train_l, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train_l + val if np.isfinite(v)]
    top = max([0.2] + [1.1 * v for v in mine])
    p = chart.Panel((0.5, max(len(val), len(train_l), 1) + 0.5), (0, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for ys, dashed, name, dy in ((train_l, False, labels["train"], 0.07), (val, True, labels["val"], -0.08)):
        if ys:
            p.line(list(range(1, len(ys) + 1)), ys, "loss", 2.2, dashed=dashed)
            p.text(len(ys) - 0.3 if len(ys) > 1 else 1, ys[-1] + dy * top, name, "loss", "end")   # clear of the line at the last epoch
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=276, height=200)   # 11 px text in a play box on a phone
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_picture(boxes: list, words: dict) -> str:
    """The test scenes of the rewind, each with its true boxes and the boxes of 0.5 or more that the learner's
    detector gave it at one moment, after NMS."""
    panels = []
    for seed, found in zip(SHOWN, boxes):
        s = scenes.scene(seed)
        panels.append((_scene_svg(s, found, words["scene"].format(seed=seed, n=len(found)), figsize=(2.3, 2.45),
                                  crop=(0, 0, 127, 127), size=9, labels=False), 160))
    return _side_by_side(panels, words["alt"])


# --- exercise 2: targets ----------------------------------------------------------------------------------------------

_T_HEAD = ("import torch\nfrom cnn.match import iou_matrix, match\nfrom cnn.offsets import offsets\n\n\n"
           "def targets(anchors, boxes, labels):\n"
           "    # anchors: (A, 4); boxes: (B, 4), each row (top, left, bottom, right); labels: (B,), the kind of each box\n"
           "    # label: for every anchor the number of the box it learns, -1 for background, -2 when it is ignored\n"
           "    label = torch.tensor(match(iou_matrix(anchors.numpy(), boxes.numpy())))\n")
_T_START = (_T_HEAD + "    # give back classes, shape (A,): 0 for background, the kind + 1 for an anchor that learns a box,\n"
            "    # -1 when it is ignored; and t, shape (A, 4): the offsets of each anchor that learns a box from that box\n"
            "    ...\n")
_T_LEARN = "    learn = label >= 0\n"
_T_ZEROS = "    classes = torch.zeros(len(anchors), dtype=torch.int64)\n"
_T_KIND = "    classes[learn] = labels[label[learn]] + 1\n"
_T_IGNORE = "    classes[label == -2] = -1\n"
_T_OFFSETS = "    t = torch.zeros(len(anchors), 4)\n    t[learn] = offsets(anchors[learn], boxes[label[learn]])\n"
_T_RETURN = "    return classes, t\n"


def _t(learn=_T_LEARN, zeros=_T_ZEROS, kind=_T_KIND, ignore=_T_IGNORE, offs=_T_OFFSETS, ret=_T_RETURN) -> str:
    return _T_HEAD + learn + zeros + kind + ignore + offs + ret


_T = _t()
_T_SCENE = ("from everything_ai.scenes import anchor_boxes, scene\n\n"
            "s = scene({seed})\nanchors = torch.tensor(anchor_boxes({anchors})).float()\n"
            "boxes = torch.tensor([shape.box for shape in s.shapes]).float()\nlabels = torch.tensor([shape.label for shape in s.shapes])\n"
            "classes, t = targets(anchors, boxes, labels)\nclasses, t = torch.as_tensor(classes), torch.as_tensor(t)\n")
_T_SCENARIOS = [
    # four anchors and two boxes: anchor 0 learns the triangle, 1 is ignored, 2 learns the circle, 3 background
    "anchors = torch.tensor([[0.0, 0.0, 15.0, 15.0], [0.0, 8.0, 15.0, 23.0], [40.0, 40.0, 71.0, 71.0], [100.0, 100.0, 115.0, 115.0]])\n"
    "boxes = torch.tensor([[0.0, 2.0, 15.0, 17.0], [42.0, 40.0, 73.0, 71.0]])\nlabels = torch.tensor([2, 0])\n"
    "classes, t = targets(anchors, boxes, labels)\nclasses, t = torch.as_tensor(classes), torch.as_tensor(t)\n[classes.tolist(), t[classes > 0].tolist()]",
    # the scene of the worked example: how many anchors of each class (-1, 0, 1, 2, 3), and the circle's best anchor
    _T_SCENE.format(seed=SEED, anchors="") + "[[classes.tolist().count(c) for c in (-1, 0, 1, 2, 3)], t[458].tolist()]",
    # another scene, with other anchors: two sizes on 8 by 8 squares of 16 pixels
    _T_SCENE.format(seed=7, anchors="cells=8, stride=16, sizes=((24, 24), (40, 40))")
    + "[[classes.tolist().count(c) for c in (-1, 0, 1, 2, 3)], t[classes > 0].tolist()]",
]
_T_WRONG = {
    "no-plus-one": _t(kind="    classes[learn] = labels[label[learn]]\n"),
    "box-number": _t(kind="    classes[learn] = label[learn] + 1\n"),
    "no-ignore": _t(ignore=""),
    "ignore-minus-two": _t(ignore="    classes[label == -2] = -2\n"),
    "background-minus-one": _t(zeros="    classes = torch.zeros(len(anchors), dtype=torch.int64) - 1\n"),
    "swapped": _t(offs="    t = torch.zeros(len(anchors), 4)\n    t[learn] = offsets(boxes[label[learn]], anchors[learn])\n"),
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"
_K_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_K_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _raised_in(error: BaseException, name: str) -> bool:
    """The error happened inside the function of that name: in the learner's own code, not in the check's lines."""
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == name:
            return True
        tb = tb.tb_next
    return False


def _close(got, want) -> bool:
    """The same value, with floats within 1 in 10,000 of each other (plus a tiny bit for 0)."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool) or math.isnan(got):
            return False
        return abs(got - want) <= 1e-4 * abs(want) + 1e-6
    if isinstance(want, int) and isinstance(got, float):
        return got == want
    return got == want


_T_MISTAKES = [
    Mistake("unpack", error=lambda e: isinstance(e, (ValueError, TypeError)) and re.search(r"values to unpack|cannot unpack|not iterable", str(e)) is not None,
            en="The check takes two things from your function, `classes, t = targets(anchors, boxes, labels)`, and your function gives back something else. Give back both, with a comma between them: `return classes, t`.",
            nl="De check haalt twee dingen uit je functie, `classes, t = targets(anchors, boxes, labels)`, en je functie geeft iets anders terug. Geef ze allebei terug, met een komma ertussen: `return classes, t`."),
    Mistake("float-classes", error=lambda e: isinstance(e, RuntimeError) and "dtypes match" in str(e),
            en="Python stops when your function puts the kinds into `classes`: `torch.zeros` gives floats, unless you say otherwise, and the kinds are whole numbers. Make `classes` whole numbers from the start, `torch.zeros(len(anchors), dtype=torch.int64)`: `F.cross_entropy` wants its classes so too.",
            nl="Python stopt als je functie de soorten in `classes` zet: `torch.zeros` geeft floats, tenzij je iets anders zegt, en de soorten zijn gehele getallen. Maak `classes` meteen gehele getallen, `torch.zeros(len(anchors), dtype=torch.int64)`: `F.cross_entropy` wil zijn klassen ook zo."),
    Mistake("fixed-length", error=lambda e: isinstance(e, IndexError) and re.search(r"indexed tensor \[768", str(e)) is not None,
            en="Your `classes` or `t` has 768 rows, as in the worked example, and the check first gives your function 4 anchors. Make them as long as the anchors it gets: `torch.zeros(len(anchors), ...)`.",
            nl="Je `classes` of `t` heeft 768 rijen, zoals in het uitgewerkte voorbeeld, en de check geeft je functie eerst 4 anchors. Maak ze zo lang als de anchors die hij krijgt: `torch.zeros(len(anchors), ...)`."),
    Mistake("no-plus-one", wrong=_T_WRONG["no-plus-one"],
            en=_S_EN + ": an anchor that learns a circle gets class 0, and 0 is background. The kinds are 0, 1 and 2, and the classes 1, 2 and 3: add 1, `labels[label[learn]] + 1`.",
            nl=_S_NL + ": een anchor die een cirkel leert, krijgt klasse 0, en 0 is achtergrond. De soorten zijn 0, 1 en 2, en de klassen 1, 2 en 3: tel er 1 bij op, `labels[label[learn]] + 1`."),
    Mistake("box-number", wrong=_T_WRONG["box-number"],
            en=_S_EN + ": you give each anchor the number of its box, plus 1. An anchor learns the kind of that box: `labels[label[learn]]` looks it up for every anchor that learns a box, and + 1 makes room for background.",
            nl=_S_NL + ": je geeft elke anchor het nummer van zijn box, plus 1. Een anchor leert de soort van die box: `labels[label[learn]]` zoekt die op voor elke anchor die een box leert, en + 1 maakt plaats voor achtergrond."),
    Mistake("no-ignore", wrong=_T_WRONG["no-ignore"],
            en=_S_EN + ": the ignored anchors get class 0, so they would learn background. They lie partly on a shape: give them -1, where `label` is -2, so that the loss can leave them out.",
            nl=_S_NL + ": de genegeerde anchors krijgen klasse 0, dus zouden ze achtergrond leren. Ze liggen deels op een vorm: geef ze -1, waar `label` -2 is, zodat de loss ze kan weglaten."),
    Mistake("ignore-minus-two", wrong=_T_WRONG["ignore-minus-two"],
            en=_S_EN + ": the ignored anchors get class -2. `match` writes -2 for them, but here a class says it: -1 for an ignored anchor, so that the loss can leave out every class below 0.",
            nl=_S_NL + ": de genegeerde anchors krijgen klasse -2. `match` schrijft -2 voor ze, maar hier zegt een klasse het: -1 voor een genegeerde anchor, zodat de loss elke klasse onder 0 kan weglaten."),
    Mistake("background-minus-one", wrong=_T_WRONG["background-minus-one"],
            en=_S_EN + ": the anchors that learn background get class -1, the class of ignored anchors. Background is a class too, class 0, and almost every anchor learns it: start from zeros.",
            nl=_S_NL + ": de anchors die achtergrond leren, krijgen klasse -1, de klasse van genegeerde anchors. Achtergrond is ook een klasse, klasse 0, en bijna elke anchor leert hem: begin met nullen."),
    Mistake("swapped", wrong=_T_WRONG["swapped"],
            en=_S_EN + ": the offsets have the wrong sign and size: `offsets` takes the anchors first, then their boxes, `offsets(anchors[learn], boxes[label[learn]])`.",
            nl=_S_NL + ": de offsets hebben het verkeerde teken en de verkeerde grootte: `offsets` neemt eerst de anchors, dan hun boxes, `offsets(anchors[learn], boxes[label[learn]])`."),
]

# --- exercise 3: detection_loss ---------------------------------------------------------------------------------------

_L_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef detection_loss(out, classes, t):\n"
           "    # out: (N, A, 8), for every anchor 4 logits (background, circle, square, triangle) and 4 offsets\n"
           "    # classes: (N, A) and t: (N, A, 4), what targets gives for each scene\n")
_L_START = (_L_HEAD + "    # give back the cross-entropy of the logits over the anchors that are not ignored, plus the mean\n"
            "    # of the absolute differences of the offsets over the anchors that learn a box\n    ...\n")
_L_KEPT = "    kept = classes >= 0\n"
_L_KIND = "    kind_loss = F.cross_entropy(out[:, :, :4][kept], classes[kept])\n"
_L_LEARN = "    learn = classes > 0\n"
_L_BOX = "    box_loss = (out[:, :, 4:][learn] - t[learn]).abs().mean()\n"
_L_RETURN = "    return kind_loss + box_loss\n"


def _l(kept=_L_KEPT, kind=_L_KIND, learn=_L_LEARN, box=_L_BOX, ret=_L_RETURN) -> str:
    return _L_HEAD + kept + kind + learn + box + ret


_L = _l()
_L_SCENARIOS = [
    # one scene of four anchors, the classes and offsets of exercise 2: a triangle, ignored, a circle, background
    "out = torch.tensor([[[2.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],\n"
    "                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], [1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]]])\n"
    "classes = torch.tensor([[3, -1, 1, 0]])\n"
    "t = torch.tensor([[[0.0, 0.125, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0625, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])\n"
    "float(detection_loss(out, classes, t))",
    # two scenes of 768 anchors, made up
    "torch.manual_seed(0)\nout = torch.randn(2, 768, 8)\nclasses = torch.randint(0, 4, (2, 768))\nclasses[:, ::5] = -1\n"
    "t = 0.2 * torch.randn(2, 768, 4)\nfloat(detection_loss(out, classes, t))",
    # the same scenes, almost all background, as real scenes are
    "torch.manual_seed(1)\nout = torch.randn(2, 768, 8)\nclasses = torch.zeros(2, 768, dtype=torch.int64)\n"
    "classes[0, 458] = 1\nclasses[1, 100:110] = 3\nclasses[:, 200:213] = -1\nt = 0.2 * torch.randn(2, 768, 4)\n"
    "float(detection_loss(out, classes, t))",
    # what training does with it: backward, back to the 8 numbers of every anchor
    "out = torch.zeros(1, 4, 8, requires_grad=True)\nclasses = torch.tensor([[3, -1, 1, 0]])\n"
    "t = torch.tensor([[[0.0, 0.125, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0625, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])\n"
    "loss = detection_loss(out, classes, t)\nloss.backward()\nfloat(out.grad.abs().sum())",
]
_L_WRONG = {
    "box-all": _l(learn="    learn = classes >= 0\n"),
    "box-sum": _l(box="    box_loss = (out[:, :, 4:][learn] - t[learn]).abs().sum()\n"),
    "squared": _l(box="    box_loss = ((out[:, :, 4:][learn] - t[learn]) ** 2).mean()\n"),
    "no-box": _l(ret="    return kind_loss\n"),
    "kind-learn": _l(kind="    kind_loss = F.cross_entropy(out[:, :, :4][learn], classes[learn])\n", learn="", kept=_L_LEARN),
    "swapped": _l(kind="    kind_loss = F.cross_entropy(out[:, :, 4:][kept], classes[kept])\n",
                  box="    box_loss = (out[:, :, :4][learn] - t[learn]).abs().mean()\n"),
    "smooth": _l(box="    box_loss = F.smooth_l1_loss(out[:, :, 4:][learn], t[learn])\n"),
    "per-anchor": _l(box="    box_loss = (out[:, :, 4:][learn] - t[learn]).abs().sum(dim=1).mean()\n"),
}
_L_MISTAKES = [
    Mistake("ignored", error=lambda e: isinstance(e, (IndexError, RuntimeError)) and re.search(r"Target -\d+ is out of bounds|out of bounds", str(e)) is not None
            and "Target" in str(e),
            en="`F.cross_entropy` stops: it got class -1, and a class must be one of 0 to 3. The ignored anchors have class -1: leave them out with a mask, `kept = classes >= 0`, and give the cross-entropy only `out[:, :, :4][kept]` and `classes[kept]`.",
            nl="`F.cross_entropy` stopt: hij kreeg klasse -1, en een klasse moet een van 0 tot 3 zijn. De genegeerde anchors hebben klasse -1: laat ze weg met een masker, `kept = classes >= 0`, en geef de cross-entropy alleen `out[:, :, :4][kept]` en `classes[kept]`."),
    Mistake("item", error=lambda e: (isinstance(e, AttributeError) and "attribute 'backward'" in str(e))
            or (isinstance(e, RuntimeError) and "does not require grad" in str(e)),
            en="`loss.backward()` stops on what your function gives back: a plain number, without the steps that made it. `.item()` and `float()` cut the loss off from training, and so does a new tensor made from it. Give back the tensor itself: `return kind_loss + box_loss`.",
            nl="`loss.backward()` stopt op wat je functie teruggeeft: een los getal, zonder de stappen waarmee het gemaakt is. `.item()` en `float()` snijden de loss los van het trainen, en een nieuwe tensor die ervan gemaakt is ook. Geef de tensor zelf terug: `return kind_loss + box_loss`."),
    Mistake("box-all", wrong=_L_WRONG["box-all"],
            en=_S_EN + ": your box loss takes the background anchors too, and pulls their offsets to 0. A background anchor has no box to learn: take the offsets only where `classes > 0`.",
            nl=_S_NL + ": je box-loss neemt de anchors met achtergrond ook mee, en trekt hun offsets naar 0. Een anchor met achtergrond heeft geen box om te leren: neem de offsets alleen waar `classes > 0`."),
    Mistake("box-sum", wrong=_L_WRONG["box-sum"],
            en=_S_EN + ": your box loss adds the differences up. Take their mean, `.mean()`: then the box loss does not grow with the number of anchors that learn a box, and stays on the scale of the kind loss, which is a mean too.",
            nl=_S_NL + ": je box-loss telt de verschillen op. Neem hun gemiddelde, `.mean()`: dan groeit de box-loss niet met het aantal anchors dat een box leert, en blijft hij op de schaal van de loss van de soorten, die ook een gemiddelde is."),
    Mistake("squared", wrong=_L_WRONG["squared"],
            en=_S_EN + ": your box loss squares the differences. Take their absolute value, `.abs()`, as the mean absolute error of [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en): a box that is far off then does not pull much harder than one that is a little off.",
            nl=_S_NL + ": je box-loss kwadrateert de verschillen. Neem hun absolute waarde, `.abs()`, zoals de mean absolute error van [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl): een box die ver ernaast zit, trekt dan niet veel harder dan een die een beetje ernaast zit."),
    Mistake("no-box", wrong=_L_WRONG["no-box"],
            en=_S_EN + ": you give back the kind loss only. Without the box loss, the detector never learns to move and stretch its anchors: add the mean absolute difference of the offsets of the anchors that learn a box.",
            nl=_S_NL + ": je geeft alleen de loss van de soorten terug. Zonder de box-loss leert de detector nooit zijn anchors te verplaatsen en uit te rekken: tel het gemiddelde absolute verschil van de offsets van de anchors die een box leren erbij op."),
    Mistake("kind-learn", wrong=_L_WRONG["kind-learn"],
            en=_S_EN + ": your cross-entropy takes only the anchors that learn a shape. Then the detector never learns background, and says a kind on every square. Take every anchor that is not ignored: `classes >= 0`.",
            nl=_S_NL + ": je cross-entropy neemt alleen de anchors die een vorm leren. Dan leert de detector nooit achtergrond, en zegt hij op elk vakje een soort. Neem elke anchor die niet genegeerd wordt: `classes >= 0`."),
    Mistake("swapped", wrong=_L_WRONG["swapped"],
            en=_S_EN + ": the logits are the first 4 numbers of every anchor, `out[:, :, :4]`, and the offsets the last 4, `out[:, :, 4:]`.",
            nl=_S_NL + ": de logits zijn de eerste 4 getallen van elke anchor, `out[:, :, :4]`, en de offsets de laatste 4, `out[:, :, 4:]`."),
    Mistake("smooth", wrong=_L_WRONG["smooth"],
            en=_S_EN + ": `F.smooth_l1_loss` squares the small differences, as SSD's loss does. This lesson takes the plain absolute difference: `(out[:, :, 4:][learn] - t[learn]).abs().mean()`, or `F.l1_loss`.",
            nl=_S_NL + ": `F.smooth_l1_loss` kwadrateert de kleine verschillen, zoals de loss van SSD doet. Deze les neemt het gewone absolute verschil: `(out[:, :, 4:][learn] - t[learn]).abs().mean()`, of `F.l1_loss`."),
    Mistake("per-anchor", wrong=_L_WRONG["per-anchor"],
            en=_S_EN + ": you add the 4 differences of every anchor up, then take the mean over the anchors: 4 times the mean of all differences. Take the mean over all of them at once, `.abs().mean()`.",
            nl=_S_NL + ": je telt de 4 verschillen van elke anchor op, en neemt dan het gemiddelde over de anchors: 4 keer het gemiddelde van alle verschillen. Neem het gemiddelde over allemaal tegelijk, `.abs().mean()`."),
]

# --- exercise 4: Detector ---------------------------------------------------------------------------------------------

_D_HEAD = ("import torch\nfrom torch import nn\n\n\nclass Detector(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
           "        # from a scene of 3 x 128 x 128 to 64 feature maps of 16 x 16: three poolings of 2\n"
           "        self.backbone = nn.Sequential(\n"
           "            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),\n"
           "            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),\n"
           "            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),\n"
           "            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),\n"
           "        )\n"
           "        # the head: a convolution of 3 by 3 that gives every square 8 numbers for each of its 3 anchors\n")
_D_HEAD_LINE = "        self.head = nn.Conv2d(64, 3 * 8, 3, padding=1)\n"
_D_FORWARD = ("\n    def forward(self, x):\n"
              "        # x: (N, 3, 128, 128); give back (N, 768, 8): one row per anchor, in the order of anchor_boxes\n")
_D_START = _D_HEAD + "        self.head = ...\n" + _D_FORWARD + "        ...\n"
_D_BODY = "        out = self.head(self.backbone(x))\n        return out.permute(0, 2, 3, 1).reshape(len(x), -1, 8)\n"


def _d(head=_D_HEAD_LINE, body=_D_BODY) -> str:
    return _D_HEAD + head + _D_FORWARD + body


_D = _d()
_D_SCENARIOS = [
    "torch.manual_seed(0)\nnet = Detector()\nx = torch.rand(2, 3, 128, 128)\nwith torch.no_grad():\n    out = net(x)\n"
    "[list(out.shape), out[0, 458].tolist()]",
    "torch.manual_seed(1)\nnet = Detector()\nx = torch.rand(3, 3, 128, 128)\nwith torch.no_grad():\n    out = net(x)\n"
    "[list(out.shape), out[2, 0].tolist(), out[2, 767].tolist(), sum(p.numel() for p in net.parameters())]",
]
_D_WRONG = {
    "no-permute": _d(body="        out = self.head(self.backbone(x))\n        return out.reshape(len(x), -1, 8)\n"),
    "columns-first": _d(body="        out = self.head(self.backbone(x))\n        return out.permute(0, 3, 2, 1).reshape(len(x), -1, 8)\n"),
    "kernel-1": _d(head="        self.head = nn.Conv2d(64, 3 * 8, 1)\n"),
    "no-padding": _d(head="        self.head = nn.Conv2d(64, 3 * 8, 3)\n"),
}
def _not_24(message: str) -> bool:
    """A reshape stopped because the head gives no 24 channels on 16 by 16 squares: its size is no whole number of
    scenes of 6,144 numbers. (A batch of a fixed number of scenes stops on a whole number of them.)"""
    m = re.search(r"shape '\[.*\]' is invalid for input of size (\d+)", message)
    return m is not None and int(m.group(1)) % (16 * 16 * 3 * 8) != 0


def _fixed_batch(message: str) -> bool:
    """A reshape to a fixed number of scenes stopped on a batch of another number of them: the input is a whole number
    of scenes of 6,144 numbers, and the shape asked for starts with another number of scenes."""
    m = re.search(r"shape '\[(\d+), .*\]' is invalid for input of size (\d+)", message)
    return m is not None and int(m.group(2)) % (16 * 16 * 3 * 8) == 0 and int(m.group(2)) != int(m.group(1)) * 16 * 16 * 3 * 8


def _rows(got) -> bool:
    """The network gives another number of rows than 768 for a scene: the first scenario starts with the shape."""
    return isinstance(got, list) and len(got) > 0 and isinstance(got[0], list) and len(got[0]) == 3 and got[0][1] != 768


_D_MISTAKES = [
    Mistake("view", error=lambda e: isinstance(e, RuntimeError) and "view size is not compatible" in str(e),
            en="`.view` stops: after `permute`, the numbers of the tensor no longer lie in memory in the order of its new axes, and `.view` cannot change a shape without moving them. Use `.reshape`, which moves them where it has to.",
            nl="`.view` stopt: na `permute` liggen de getallen van de tensor in het geheugen niet meer in de volgorde van zijn nieuwe axes, en `.view` kan een shape niet veranderen zonder ze te verplaatsen. Gebruik `.reshape`, dat ze verplaatst waar het moet."),
    Mistake("fixed-batch", error=lambda e: isinstance(e, RuntimeError) and _fixed_batch(str(e)),
            en="The reshape stops: it has room for a fixed number of scenes, such as the 1 of the worked example, and the check gives your network 2 scenes at once, then 3. Take the number of scenes of the batch that comes in: `reshape(len(x), 768, 8)`.",
            nl="De reshape stopt: hij heeft plaats voor een vast aantal scènes, zoals de 1 van het uitgewerkte voorbeeld, en de check geeft je netwerk 2 scènes tegelijk, en dan 3. Neem het aantal scènes van de batch die binnenkomt: `reshape(len(x), 768, 8)`."),
    Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and _not_24(str(e)),
            en="The reshape stops: the head gives another number of channels than 3 anchors times 8 numbers. Every square needs 24 channels: `nn.Conv2d(64, 3 * 8, 3, padding=1)`, so that 16 x 16 squares of 24 numbers make 768 rows of 8.",
            nl="De reshape stopt: de head geeft een ander aantal channels dan 3 anchors keer 8 getallen. Elk vakje heeft 24 channels nodig: `nn.Conv2d(64, 3 * 8, 3, padding=1)`, zodat 16 x 16 vakjes van 24 getallen 768 rijen van 8 maken."),
    Mistake("no-permute", wrong=_D_WRONG["no-permute"],
            en=_K_EN + ": the shape is right, but the rows do not belong to the anchors. The head gives (N, 24, 16, 16): a reshape alone takes 8 numbers of one channel, from 8 squares next to each other. First `permute(0, 2, 3, 1)` to (N, 16, 16, 24), so that the 24 numbers of a square lie together, then reshape.",
            nl=_K_NL + ": de shape klopt, maar de rijen horen niet bij de anchors. De head geeft (N, 24, 16, 16): een reshape alleen neemt 8 getallen van één channel, van 8 vakjes naast elkaar. Eerst `permute(0, 2, 3, 1)` naar (N, 16, 16, 24), zodat de 24 getallen van een vakje bij elkaar liggen, dan reshape."),
    Mistake("columns-first", wrong=_D_WRONG["columns-first"],
            en=_K_EN + ": your rows go column by column. The anchors go row by row, as `anchor_boxes` lays them: `permute(0, 2, 3, 1)` keeps the rows, axis 2, before the columns, axis 3.",
            nl=_K_NL + ": je rijen gaan kolom voor kolom. De anchors gaan rij voor rij, zoals `anchor_boxes` ze legt: `permute(0, 2, 3, 1)` houdt de rijen, axis 2, vóór de kolommen, axis 3."),
    Mistake("kernel-1", wrong=_D_WRONG["kernel-1"],
            en=_K_EN + ": your head is a convolution of 1 by 1. Take 3 by 3, with `padding=1`: then every square also sees the squares around it, and a shape that sticks out of its square.",
            nl=_K_NL + ": je head is een convolutie van 1 bij 1. Neem 3 bij 3, met `padding=1`: dan ziet elk vakje ook de vakjes eromheen, en een vorm die buiten zijn vakje uitsteekt."),
    Mistake("no-padding", wrong=_D_WRONG["no-padding"],
            en=_K_EN + ": without padding, a convolution of 3 by 3 makes 16 x 16 into 14 x 14, and the squares at the edge lose their anchors. Add `padding=1`.",
            nl=_K_NL + ": zonder padding maakt een convolutie van 3 bij 3 van 16 x 16 een 14 x 14, en verliezen de vakjes aan de rand hun anchors. Voeg `padding=1` toe."),
    Mistake("rows", matches=lambda got, text: _rows(got),
            en=_K_EN + ": your network gives another number of rows than 768, one per anchor. The head must give 3 × 8 = 24 channels on each of the 16 × 16 squares: `nn.Conv2d(64, 3 * 8, 3, padding=1)`.",
            nl=_K_NL + ": je netwerk geeft een ander aantal rijen dan 768, één per anchor. De head moet 3 × 8 = 24 channels geven op elk van de 16 × 16 vakjes: `nn.Conv2d(64, 3 * 8, 3, padding=1)`."),
]

EXERCISES = {
    "targets": ScenarioBox(
        name="targets",
        start=_T_START,
        solution=_T,
        scenarios=_T_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "targets"),
        first=[_DOTS],
        mistakes=_T_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: `classes`, 0 everywhere, the kind + 1 where `label` names a box, -1 where it is -2; `t`, the offsets of the anchors that learn a box from their boxes; and give back both.",
                      "nl": "Vervang de `...` door je code: `classes`, overal 0, de soort + 1 waar `label` een box noemt, -1 waar hij -2 is; `t`, de offsets van de anchors die een box leren vanaf hun boxes; en geef ze allebei terug."},
            "wrong": {"en": _S_EN + ": the class of every anchor (for a scene: how many anchors have each class, -1 to 3), then the offsets of the anchors that learn a box. Is a class 0 for background, the kind of the box + 1 for an anchor that learns it, and -1 for an ignored anchor? And are the offsets those of each anchor that learns a box, from its own box?",
                      "nl": _S_NL + ": de klasse van elke anchor (voor een scène: hoeveel anchors elke klasse hebben, -1 tot 3), en dan de offsets van de anchors die een box leren. Is een klasse 0 voor achtergrond, de soort van de box + 1 voor een anchor die hem leert, en -1 voor een genegeerde anchor? En zijn de offsets die van elke anchor die een box leert, vanaf zijn eigen box?"},
            "almost": {"en": "Right for the four anchors. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of anchors and boxes?",
                       "nl": "Goed voor de vier anchors. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal anchors en boxes?"},
            "correct": {"en": "Your `targets` is right: on the scene, 735 anchors learn background, 20 a shape, and 13 are ignored. It is now in your project, as `{saved}`.",
                        "nl": "Je `targets` klopt: in de scène leren 735 anchors achtergrond, 20 een vorm, en worden er 13 genegeerd. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `targets` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `targets` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "detection-loss": ScenarioBox(
        name="detection-loss",
        start=_L_START,
        solution=_L,
        scenarios=_L_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "detection_loss"),
        first=[_DOTS],
        mistakes=_L_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the cross-entropy of `out[:, :, :4]` over the anchors with a class of 0 or more, plus the mean absolute difference of `out[:, :, 4:]` and `t` over the anchors with a class above 0.",
                      "nl": "Vervang de `...` door je code: de cross-entropy van `out[:, :, :4]` over de anchors met een klasse van 0 of meer, plus het gemiddelde absolute verschil van `out[:, :, 4:]` en `t` over de anchors met een klasse boven 0."},
            "wrong": {"en": _S_EN + ". Does the cross-entropy take every anchor with a class of 0 or more, and the box loss the mean absolute difference of the offsets of the anchors with a class above 0? And do you add the two?",
                      "nl": _S_NL + ". Neemt de cross-entropy elke anchor met een klasse van 0 of meer, en de box-loss het gemiddelde absolute verschil van de offsets van de anchors met een klasse boven 0? En tel je de twee op?"},
            "almost": {"en": "Right for one scene of four anchors. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for a batch of scenes, of any number of anchors?",
                       "nl": "Goed voor één scène van vier anchors. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor een batch van scènes, met elk aantal anchors?"},
            "correct": {"en": "Your `detection_loss` is right: one loss for the kinds, over every anchor that is not ignored, and one for the boxes, over the anchors that learn one. It is now in your project, as `{saved}`.",
                        "nl": "Je `detection_loss` klopt: één loss voor de soorten, over elke anchor die niet genegeerd wordt, en één voor de boxes, over de anchors die er een leren. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `detection_loss` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `detection_loss` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "detector": ClassBox(
        name="detector",
        picture=netview.network("Detector", lambda: yours.scene_batch(128), chosen=0),
        start=_D_START,
        solution=_D,
        scenarios=_D_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "detector"),
        first=[_DOTS],
        mistakes=_D_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the head, a convolution of 3 by 3 from 64 channels to 3 times 8; and `forward`, the backbone and the head, then the 24 numbers of every square as 3 rows of 8.",
                      "nl": "Vervang de twee `...`: de head, een convolutie van 3 bij 3 van 64 channels naar 3 keer 8; en `forward`, de backbone en de head, en dan de 24 getallen van elk vakje als 3 rijen van 8."},
            "wrong": {"en": _K_EN + ": the shape of what your network gives, then the row of one anchor. Does the head give 24 channels with a convolution of 3 by 3 and `padding=1`, and do the rows go square by square, row by row, with the 3 anchors of a square after each other?",
                      "nl": _K_NL + ": de shape van wat je netwerk geeft, en dan de rij van één anchor. Geeft de head 24 channels met een convolutie van 3 bij 3 en `padding=1`, en gaan de rijen vakje voor vakje, rij voor rij, met de 3 anchors van een vakje na elkaar?"},
            "almost": {"en": "Right for 2 scenes. But " + _K_EN[0].lower() + _K_EN[1:] + ": the shape, the rows of the first and the last anchor of the last scene, and the number of parameters. Does your network work for any number of scenes?",
                       "nl": "Goed voor 2 scènes. Maar " + _K_NL[0].lower() + _K_NL[1:] + ": de shape, de rijen van de eerste en de laatste anchor van de laatste scène, en het aantal parameters. Werkt je netwerk voor elk aantal scènes?"},
            "correct": {"en": "Your `Detector` is right: for every scene, 768 rows of 8, one per anchor, in the order of `anchor_boxes`, with 38,616 parameters. It is now in your project, as `{saved}`.",
                        "nl": "Je `Detector` klopt: voor elke scène 768 rijen van 8, één per anchor, in de volgorde van `anchor_boxes`, met 38.616 parameters. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `Detector` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Detector` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "head-size" is the first choice, "why-nms" the open question.
REQUIRED = ["head-size", "targets", "detection-loss", "detector", "why-nms"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("targets", _T_START, "empty", None),
    ("targets", _T, "correct", None),
    ("targets", _T_START.replace("    ...\n", _T_LEARN + _T_ZEROS + _T_KIND + _T_IGNORE + _T_OFFSETS + _T_RETURN), "correct", None),
    # offsets for every anchor, and numbers that are not a tensor of whole numbers: only what the loss uses counts
    ("targets", _t(offs="    t = offsets(anchors, boxes[label.clamp(min=0)])\n", ret="    return classes.float(), t\n"), "correct", None),
    # a loop over the anchors
    ("targets", _t(learn="", zeros="", kind="", ignore="",
                   offs="    classes = []\n    rows = []\n    for i in range(len(anchors)):\n        if label[i] >= 0:\n"
                        "            classes.append(int(labels[label[i]]) + 1)\n            rows.append(offsets(anchors[i:i + 1], boxes[label[i]:label[i] + 1])[0])\n"
                        "        else:\n            classes.append(0 if label[i] == -1 else -1)\n            rows.append(torch.zeros(4))\n"
                        "    classes = torch.tensor(classes)\n    t = torch.stack(rows)\n"), "correct", None),
    ("targets", _t(zeros="    classes = torch.zeros(len(anchors))\n"), "error", "float-classes"),
    # 768 rows, as in the worked example
    ("targets", _t(zeros="    classes = torch.zeros(768, dtype=torch.int64)\n"), "error", "fixed-length"),
    ("targets", _t(offs="    t = torch.zeros(768, 4)\n    t[learn] = offsets(anchors[learn], boxes[label[learn]])\n"), "error", "fixed-length"),
    # lists, not tensors: never wrong
    ("targets", _t(ret="    return classes.tolist(), t.tolist()\n"), "correct", None),
    ("targets", _t(ret="    return classes\n"), "error", "unpack"),
    ("targets", _t(ret=""), "error", "unpack"),
    ("targets", _t(kind="    classes[learn] = ...\n"), "wrong", "dots-left"),
    # right for the four anchors only
    ("targets", _T_HEAD + "    if len(anchors) == 4:\n" + "".join("    " + line + "\n" for line in (_T_LEARN + _T_ZEROS + _T_KIND + _T_IGNORE + _T_OFFSETS + _T_RETURN).splitlines())
     + "    return torch.zeros(len(anchors), dtype=torch.int64), torch.zeros(len(anchors), 4)\n", "almost", None),
    ("targets", _t(kind="", ignore=""), "wrong", None),
] + [("targets", code, "wrong", key) for key, code in _T_WRONG.items()] + [
    ("detection-loss", _L_START, "empty", None),
    ("detection-loss", _L, "correct", None),
    ("detection-loss", _L_START.replace("    ...\n", _L_KEPT + _L_KIND + _L_LEARN + _L_BOX + _L_RETURN), "correct", None),
    # F.l1_loss, and the cross-entropy with ignore_index over all anchors at once
    ("detection-loss", _l(kept="", kind="    kind_loss = F.cross_entropy(out[:, :, :4].reshape(-1, 4), classes.reshape(-1), ignore_index=-1)\n",
                          box="    box_loss = F.l1_loss(out[:, :, 4:][learn], t[learn])\n"), "correct", None),
    ("detection-loss", _l(kept="", kind="    kind_loss = F.cross_entropy(out[:, :, :4].reshape(-1, 4), classes.reshape(-1))\n"), "error", "ignored"),
    ("detection-loss", _l(ret="    total = kind_loss + box_loss\n"), "error", None),
    # a plain number: training cannot follow it back
    ("detection-loss", _l(ret="    return (kind_loss + box_loss).item()\n"), "error", "item"),
    ("detection-loss", _l(ret="    return float(kind_loss + box_loss)\n"), "error", "item"),
    ("detection-loss", _l(ret="    return torch.tensor(float(kind_loss + box_loss))\n"), "error", "item"),
    ("detection-loss", _l(box="    box_loss = ...\n"), "wrong", "dots-left"),
    # right for one scene only
    ("detection-loss", _l(kept="    out, classes, t = out[:1], classes[:1], t[:1]\n" + _L_KEPT,
                          box="    box_loss = (out[:, :, 4:][learn] - t[learn]).abs().mean()\n"), "almost", None),
] + [("detection-loss", code, "wrong", key) for key, code in _L_WRONG.items()] + [
    ("detector", _D_START, "empty", None),
    ("detector", _D, "correct", None),
    ("detector", _d(body="        maps = self.head(self.backbone(x))\n        n = maps.shape[0]\n"
                         "        return maps.permute(0, 2, 3, 1).reshape(n, 16 * 16 * 3, 8)\n"), "correct", None),
    ("detector", _d(body="        out = self.head(self.backbone(x))\n        return out.permute(0, 2, 3, 1).view(len(x), -1, 8)\n"), "error", "view"),
    ("detector", _d(head="        self.head = nn.Conv2d(64, 3 * 7, 3, padding=1)\n",
                    body="        out = self.head(self.backbone(x))\n        return out.permute(0, 2, 3, 1).reshape(len(x), 768, 8)\n"), "error", "channels"),
    ("detector", _d(head="        self.head = nn.Conv2d(64, 3 * 7, 3, padding=1)\n"), "wrong", "rows"),
    ("detector", _d(head="        self.head = ...\n"), "wrong", "dots-left"),
    # right for 2 scenes only: the second scenario has 3; and the 1 scene of the worked example
    ("detector", _d(body="        out = self.head(self.backbone(x))\n        return out.permute(0, 2, 3, 1).reshape(2, 768, 8)\n"), "error", "fixed-batch"),
    ("detector", _d(body="        out = self.head(self.backbone(x))\n        return out.permute(0, 2, 3, 1).reshape(1, 768, 8)\n"), "error", "fixed-batch"),
] + [("detector", code, "wrong", key) for key, code in _D_WRONG.items()]


# --- the demo: the learner's detector, trained on the course's scenes -------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _demo(show=None):
    """For the demo: None when targets.py, detection_loss.py or detector.py is not in the project yet; "targets" when
    the learner's targets stops with an error or does not give 768 classes and offsets for the scene; "detector" when
    their Detector does, on 2 scenes; "loss" when their detection_loss does not give one finite number; "offsets" when
    their boxes_from stops with an error after training; "slow" when the training takes too long; else a dict with
    the loss after every epoch ("curves"), their boxes on the SHOWN scenes after every epoch ("moments"), the summary of
    their trained detector on the test scenes, and the modules the course's versions stood in for. `show` hears the run
    while it trains."""
    if not (_saved("targets") and _saved("detection_loss") and _saved("detector")):
        return None
    names = {}
    for module, name, key in (("targets", "targets", "targets"), ("detection_loss", "detection_loss", "loss"),
                              ("detector", "Detector", "detector"), ("offsets", "boxes_from", "offsets")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    train_ss, X = scene_set("train")
    test_ss, Xt = scene_set("test")
    try:
        C, T = with_time_limit(lambda: all_targets(train_ss, names["targets"]), TARGETS_LIMIT)
        Ct, Tt = all_targets(test_ss, names["targets"])
    except Timeout:
        return "slow"
    except Exception:
        return "targets"
    if not (C.shape == (N_TRAIN, len(anchors())) and T.shape == (N_TRAIN, len(anchors()), 4)):
        return "targets"
    try:
        torch.manual_seed(0)
        model = names["Detector"]()
        with torch.no_grad():
            if tuple(model(X[:2].float() / 255).shape) != (2, len(anchors()), PER):
                return "detector"
    except Exception:
        return "detector"
    try:
        model.train()
        loss = names["detection_loss"](model(X[:2].float() / 255), C[:2], T[:2])
        if not (torch.is_tensor(loss) and loss.numel() == 1 and torch.isfinite(loss)):
            return "loss"
    except Exception:
        return "loss"
    torch.manual_seed(0)   # the same start as the course's run: the check above drew no random numbers, but be sure
    model = names["Detector"]()
    moments, broken = [], []

    def heard(p):
        if "out" in p:
            try:
                moments.append([shown_boxes(p["out"][TEST.index(seed)], boxes_fn=names["boxes_from"]) for seed in SHOWN])
            except Exception:
                broken.append(True)
                moments.append([[] for _ in SHOWN])
            p = {k: v for k, v in p.items() if k != "out"}
        if show is not None:
            try:
                show(p)
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass

    try:
        history = with_time_limit(lambda: train(model, names["detection_loss"], (X, C, T), (Xt, Ct, Tt), 0, EPOCHS, heard), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except Exception:
        return "loss"
    if broken:
        return "offsets"
    try:
        result = summary(outputs(model, Xt), test_ss, boxes_fn=names["boxes_from"])
    except Exception:
        return "offsets"
    curves = {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]}
    return result | {"curves": curves, "moments": moments, "params": sum(v.numel() for v in model.parameters()), "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("targets", "detection_loss", "detector", "match", "offsets"):
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
