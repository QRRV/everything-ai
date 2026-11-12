# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-mask-rcnn: bilinear, the bilinear value of feature maps at every point of a grid; roi_align,
which cuts the feature maps of one box to k by k numbers per channel without rounding (RoIAlign); and paste, which
puts the small mask the mask head draws for a box back into the scene. All three are saved in the learner's project
(work/projects/cnn/bilinear.py, roi_align.py and paste.py) when right. The demo trains the course's Mask R-CNN with
them on the course's scenes: the two-stage detector of cv-rcnn, with RoIAlign instead of RoI pooling, and a mask head
that draws a mask of 16 by 16 for every kind inside every box; then it finds the shapes of new scenes with the
learner's proposals and boxes_from, pastes their masks, and measures the mask AP. And the pictures: a class map
next to the masks of every shape, the course's trained Mask R-CNN on the scene of the worked example, RoI pooling
and RoIAlign of one box, and for the slider, the masks and the mask AP of the course's Mask R-CNN trained with each
at every IoU threshold. Everything is computed from the course's own scenes (scenes.py): no download."""
import functools
import html
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch import nn

from everything_ai import chart, live, netview, paths, projects, scenes
from everything_ai.check import Mistake, ScenarioBox, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_mask_rcnn",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The two-stage detector, its training and its proposals; through it the scenes, the anchors, the targets of the
# RPN and the pictures of boxes on scenes of cv-one-stage-detector
_TWO = _lesson_checks("cv-rcnn")
_ONE = _TWO._ONE
# The class maps of a scene and their pictures, and pack/unpack for a mask as short text
_FCN = _lesson_checks("cv-fcn")

# The demo makes proposals with the learner's proposals of cv-rcnn, which imports their boxes_from of cv-anchors, and
# their roi_align and paste import their bilinear of this lesson. Where one is missing or broken, the course's version
# stands in, and the lesson says so.
BUILDS_ON = (("cnn", "offsets", "cv-anchors", "boxes-from"), ("cnn", "proposals", "cv-rcnn", "proposals"),
             ("cnn", "bilinear", "cv-mask-rcnn", "bilinear"))
projects.builds_on(*BUILDS_ON)

SEED, KINDS, STRIDE = _ONE.SEED, _ONE.KINDS, _ONE.STRIDE
N_TRAIN, N_TEST, TRAIN, TEST = _ONE.N_TRAIN, _ONE.N_TEST, _ONE.TRAIN, _ONE.TEST
EPOCHS, LR, BATCH = _ONE.EPOCHS, _ONE.LR, _ONE.BATCH
SCORE, NMS_IOU, LOW, AP_IOU = _ONE.SCORE, _ONE.NMS_IOU, _ONE.LOW, _ONE.AP_IOU
SHOWN, SHOWN_SHAPES = _ONE.SHOWN, _ONE.SHOWN_SHAPES
anchors, example, scene_set, boxes_and_labels = _ONE.anchors, _ONE.example, _ONE.scene_set, _ONE.boxes_and_labels
iou, lr_factor, offsets, boxes_from, iou_matrix = _ONE.iou, _ONE.lr_factor, _ONE.offsets, _ONE.boxes_from, _ONE.iou_matrix
number, whole = _ONE.number, _ONE.whole
proposals, as_boxes, rpn_loss, second_loss, rpn_targets = (_TWO.proposals, _TWO.as_boxes, _TWO.rpn_loss,
                                                            _TWO.second_loss, _TWO.rpn_targets)
roi_pool = _TWO.roi_pool
PER_RPN, PER, HIDDEN, FG_IOU = _TWO.PER_RPN, _TWO.PER, _TWO.HIDDEN, _TWO.FG_IOU
TRAIN_KEEP = TEST_KEEP = _TWO.TRAIN_KEEP

K = 4                         # the box head cuts every proposal to 4 by 4 bins, as in cv-rcnn
KM = 8                        # the mask head cuts every box to 8 by 8 bins ...
M = 2 * KM                    # ... and draws a mask of 16 by 16 for every kind inside it
MASK_CHANNELS = 32            # the channels of the convolutions of the mask head
SIZE = scenes.SIZE
TRAIN_LIMIT = 1500            # seconds the demo gives the training: the note says twenty-five minutes
CIRCLE = 3                    # the circle of the scene of the worked example: the box of the worked example
THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))   # the slider: the IoU a mask must be above, as COCO
INSTANCE_SEED = 104           # a test scene with two circles that touch, for the picture of a mask per object


# >>> measured
# Measured by measure(): the worked example, and what the lesson says of the masks of the test scenes
REFERENCE = {'maps': (64, 16, 16),
 'box': (58, 50, 86, 78),
 'size': (29, 29),
 'first': (6.9766, 5.9766),
 'last': (10.1484, 9.1484),
 'pool_squares': (7, 6, 10, 9),
 'aligned': (64, 8, 8),
 'row': (0.6619, 0.8114, 0.9698, 1.2657, 1.6095, 1.4621, 0.8869, 0.4258),
 'same_as_roi_align': True,
 'target': 'eNpjYA1hYAwNdWANDQ0QDQ0NgWIMIArBYHmQWpAeBtEQBgANzBBb',
 'ones': 205,
 'same_as_target': True,
 'logits': (3, 16, 16),
 'p': (0.4536, 0.4695),
 'loss': 0.7415,
 'same_as_loss': True,
 'params': 199325,
 'mask_params': 31939,
 'touching': {'shapes': 808, 'joined': 186, 'scenes': 82},
 'box_share': [0.7807, 0.5855, 0.4965],
 'round_trip': {4: 0.7734, 8: 0.8819, 14: 0.93, 16: 0.9414, 28: 0.9929}}

# Measured by measure_trained(): the course's Mask R-CNN trained with the seeds 0, 1, 2, 3, 4, with RoIAlign (the lists) and with RoI
# pooling ("pool"); with seed 0, its masks on the scene of the worked example and its mask AP at every threshold
TRAINED = {'map': [0.9409, 0.9453, 0.9459, 0.9473, 0.9477],
 'aps': [[0.9451, 0.9488, 0.9287], [0.9412, 0.9501, 0.9445], [0.9527, 0.9459, 0.9393], [0.9431, 0.9516, 0.9472],
         [0.9569, 0.9466, 0.9398]],
 'map75': [0.8505, 0.8457, 0.857, 0.8473, 0.8475],
 'box_map': [0.9614, 0.968, 0.9685, 0.9752, 0.9656],
 'box_map75': [0.8794, 0.8975, 0.8949, 0.8954, 0.9025],
 'found': [768, 770, 771, 773, 775],
 'false': [37, 44, 46, 40, 43],
 'median': [[0.9458, 0.9083, 0.9028], [0.9475, 0.9117, 0.8882], [0.9466, 0.9162, 0.8994], [0.9454, 0.9088, 0.8963],
            [0.9458, 0.9063, 0.8886]],
 'shapes': 808,
 'curves': {'rpn': [0.2376, 0.0972, 0.0828, 0.0734, 0.0646, 0.0587, 0.0526, 0.0488],
            'second': [1.0024, 0.5622, 0.3866, 0.333, 0.3016, 0.2658, 0.2346, 0.2196],
            'mask': [0.5026, 0.2908, 0.2543, 0.2379, 0.2227, 0.2114, 0.2033, 0.1999]},
 'scene': [(2, 1.0, (51.1, 7.3, 75.5, 29.0),
            'eNrt1LENwDAIRFEoGIF9MgIN+69iT3A/TaQU99snW0ZCjnDOOee+qh/tqz3BC7zBV3uCF/hqL/DWfl8nvcAbfN+cHz2+9AYP8gJP8HvBwH6CF3iO/wDnnHN/6gB9ixof'),
           (0, 0.999, (58.0, 49.4, 86.5, 78.8),
            'eNrt0rsNwCAMhGFTeATvwwgU3P6rJPIAZwRCKXJ/+4mHwGZKKaWU+riGSR1AJ+yvT74cYKeDHuDpY9ujcIBfsPB22b3wOPTF+41dtzXvxf/ZtucDVvPJ5jvo9rmBHblSSv2mByfpNEw='),
           (1, 0.957, (41.0, 40.5, 68.0, 68.6),
            'eNrt0rERACAIBEEMLMF+KEED+m9FG/A/dZzb9BgSiAAAGEPnVlP2XofrdV/ietF/6Knun+p/YukHjJm6m2y7HwAAAC/Y93Excg=='),
           (2, 0.888, (101.7, 9.0, 117.6, 22.6),
            'eNrt0qENADAIRFEqOgL7MAKi7L9KN+AEoknzn/3mSDADAAAAgG9kn9cRvVQP0VP0fsCuUj0m3asfOO/9/Z686DsXR1gJuw==')],
 'pool': {'map': [0.9245, 0.915, 0.9254, 0.9238, 0.9243],
          'aps': [[0.9418, 0.9249, 0.9067], [0.9418, 0.9092, 0.894], [0.9491, 0.9133, 0.914], [0.9548, 0.9136, 0.903],
                  [0.9452, 0.9176, 0.91]],
          'map75': [0.5352, 0.5599, 0.5922, 0.5869, 0.5837],
          'box_map': [0.955, 0.9619, 0.9569, 0.9617, 0.9624],
          'box_map75': [0.7307, 0.766, 0.7383, 0.7499, 0.7737],
          'found': [765, 766, 768, 766, 767],
          'false': [114, 102, 108, 112, 92],
          'median': [[0.8833, 0.7959, 0.729], [0.8817, 0.8052, 0.7389], [0.8874, 0.809, 0.7418],
                     [0.8873, 0.8026, 0.7484], [0.8891, 0.8179, 0.7293]]},
 'pool_scene': [(0, 0.994, (57.7, 48.4, 87.6, 78.7),
                 'eNrt07sNwCAMRVFSeAT2YQSKvP1XQSFpYyOsNNE97QUkfqUAAICkevpdau50+QOuLnd5f4HZ+3a32V+3eOj2Vbegi57qde38+25/7rcFPXj+zg8zLfyfkugA8B8DBBlAnA=='),
                (2, 0.977, (51.8, 8.2, 74.6, 29.5),
                 'eNrt0bEJwDAMRFG50AjaxyO40f6rxGSAO3BISPF/+0ASdgQREdFf6ym5jLf2NN7a03gZ39xx7qP1/DReL7vb7+6/FyzzAMv8r/Q9QXuU8ZgPnYiI6NMuryQaLQ=='),
                (1, 0.948, (44.5, 38.2, 72.3, 64.9),
                 'eNrt0zEOgDAMBMGjSE+Tf6WI//+VAD2+CBQhod12LDeWJSIi+rYtmvGeeolIN5yeDVzen3ud8XDelnngB++537PMeadcL7waL8blXObBiIiIftMAJSIxNw=='),
                (2, 0.904, (100.9, 8.3, 117.9, 24.6),
                 'eNrt0rENACAIRFEsHMF9GMHC238VNzgKQ+V/7Q8JJEQAAAAAwC/GLvrxfeq1Z9H9gkuqenZ2yS+oct7fL1600QWbJwoo'),
                (1, 0.687, (49.0, 42.8, 83.6, 75.8),
                 'eNrt07sNwCAMhGFTMAL7uKF3we2/ShSRFl+UBCnF/e0nnpLNlFJKqb8HEAYGcTz3kvtk+MrrPY+t3sn5na6P9P3cnfxfEB+7/Pq/5QXZemPeiFfihfg8IIi75RcgA5bN17mBpxPawl65UkqpjzoAFr08Tw==')],
 'thresholds': {'align': {0.5: 0.9409,
                          0.55: 0.9275,
                          0.6: 0.9088,
                          0.65: 0.893,
                          0.7: 0.8735,
                          0.75: 0.8505,
                          0.8: 0.8054,
                          0.85: 0.7395,
                          0.9: 0.5212,
                          0.95: 0.1176},
                'pool': {0.5: 0.9245,
                         0.55: 0.8936,
                         0.6: 0.8433,
                         0.65: 0.7844,
                         0.7: 0.6908,
                         0.75: 0.5352,
                         0.8: 0.3747,
                         0.85: 0.2379,
                         0.9: 0.0825,
                         0.95: 0.0006}}}
# <<< measured


# --- the course's versions --------------------------------------------------------------------------------------------

def bilinear(maps, ys, xs):
    """The bilinear value of maps (C, H, W) at every point (y, x) of the grid ys by xs, counted in squares from the
    middle of square 0, as the learner's: a point outside the maps takes the value at the nearest edge. (C, len(ys),
    len(xs))."""
    h, w = maps.shape[1], maps.shape[2]
    ys = [min(max(y, 0), h - 1) for y in ys]
    xs = [min(max(x, 0), w - 1) for x in xs]
    r0 = [int(y) for y in ys]
    r1 = [min(r + 1, h - 1) for r in r0]
    c0 = [int(x) for x in xs]
    c1 = [min(c + 1, w - 1) for c in c0]
    t = (torch.tensor(ys) - torch.tensor(r0)).reshape(-1, 1)
    u = torch.tensor(xs) - torch.tensor(c0)
    top = (1 - u) * maps[:, r0][:, :, c0] + u * maps[:, r0][:, :, c1]
    bottom = (1 - u) * maps[:, r1][:, :, c0] + u * maps[:, r1][:, :, c1]
    return (1 - t) * top + t * bottom


def bin_middles(start, length, k: int) -> list:
    """The middles of k bins of equal length along a box edge that starts at the pixel `start` and is `length` pixels
    long, both ends included: in pixels."""
    return [start - 0.5 + (i + 0.5) * length / k for i in range(k)]


def roi_align(maps, box, k: int = K):
    """RoIAlign, as the learner's: the bilinear value of the feature maps (C, H, W) at the middle of each of k by k
    bins of `box` (top, left, bottom, right, in pixels, both ends included), with every square STRIDE by STRIDE pixels
    and its middle at pixel STRIDE s + 3.5; no rounding. One point per bin. (C, k, k)."""
    top, left, bottom, right = (float(v) for v in box)
    ys = bin_middles(top, bottom - top + 1, k)
    xs = bin_middles(left, right - left + 1, k)
    return bilinear(maps, [(y - 3.5) / 8 for y in ys], [(x - 3.5) / 8 for x in xs])


def paste(mask, box, size: int = SIZE):
    """The mask of one box in the scene, as the learner's: the m by m probabilities `mask`, made as large as the box
    with bilinear interpolation, True where 0.5 or more, for every pixel whose middle lies in the box and in the
    scene. A bool tensor (size, size)."""
    top, left, bottom, right = (float(v) for v in box)
    m = mask.shape[0]
    out = torch.zeros(size, size, dtype=torch.bool)
    r0, r1 = max(math.ceil(top - 0.5), 0), min(math.floor(bottom + 0.5), size - 1)
    c0, c1 = max(math.ceil(left - 0.5), 0), min(math.floor(right + 0.5), size - 1)
    if r1 < r0 or c1 < c0:   # a box outside the scene, or one of no pixels: the mask stays empty
        return out
    ys = [(r - top + 0.5) / (bottom - top + 1) * m - 0.5 for r in range(r0, r1 + 1)]
    xs = [(c - left + 0.5) / (right - left + 1) * m - 0.5 for c in range(c0, c1 + 1)]
    out[r0:r1 + 1, c0:c1 + 1] = bilinear(mask.unsqueeze(0), ys, xs)[0] >= 0.5
    return out


def target(mask, box, m: int = M) -> torch.Tensor:
    """What the mask head learns for one box: the true mask of its shape (bool, SIZE by SIZE) read at the middle of
    each of m by m bins of the box, the pixel that middle lies in; outside the scene, background. Floats 0 and 1,
    (m, m)."""
    top, left, bottom, right = (float(v) for v in box)
    rows = [int(math.floor(y + 0.5)) for y in bin_middles(top, bottom - top + 1, m)]
    cols = [int(math.floor(x + 0.5)) for x in bin_middles(left, right - left + 1, m)]
    mask = np.asarray(mask, dtype=bool)
    inside_r = np.array([0 <= r < mask.shape[0] for r in rows])
    inside_c = np.array([0 <= c < mask.shape[1] for c in cols])
    got = mask[np.clip(rows, 0, mask.shape[0] - 1)][:, np.clip(cols, 0, mask.shape[1] - 1)]
    return torch.tensor(got & inside_r[:, None] & inside_c[None, :], dtype=torch.float32)


class MaskRCNN(nn.Module):
    """The course's Mask R-CNN: the two-stage detector of cv-rcnn (its backbone, its region proposal network and its
    second stage), and a mask head, a small FCN on the 64 x 8 x 8 numbers of a box: two 3×3 convolutions of 32
    channels with ReLU, a transposed convolution of 2×2 with stride 2 to 16 by 16 with ReLU, and a 1×1 convolution to
    one mask of logits for every kind."""

    def __init__(self):
        super().__init__()
        two = _TWO.TwoStage(K)
        self.backbone, self.rpn, self.second = two.backbone, two.rpn, two.second
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, MASK_CHANNELS, 3, padding=1), nn.ReLU(),
            nn.Conv2d(MASK_CHANNELS, MASK_CHANNELS, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(MASK_CHANNELS, MASK_CHANNELS, 2, stride=2), nn.ReLU(),
            nn.Conv2d(MASK_CHANNELS, KINDS, 1),
        )

    def forward(self, x):
        """The feature maps (N, 64, 16, 16) and the RPN's rows (N, 768, 6) of a batch of scenes."""
        maps = self.backbone(x)
        return maps, self.rpn(maps).permute(0, 2, 3, 1).reshape(len(x), -1, PER_RPN)


def pooled(maps, rois, pool_fn=roi_align, k: int = K) -> torch.Tensor:
    """The pooled features of every (scene, box) in rois, the boxes of one scene after each other: (R, 64, k, k)."""
    out, last, mine = [], None, None
    for i, box in rois:
        if i != last:
            mine, last = maps[i], i
        out.append(pool_fn(mine, box, k))
    return torch.stack(out)


def mask_loss(logits, kinds, targets) -> torch.Tensor:
    """The loss of the mask head: of the masks of every box (R, KINDS, M, M), only the mask of the kind of the true
    box it learns counts; the mean log loss of its pixels, with the sigmoid of every logit as the probability."""
    mine = logits[torch.arange(len(kinds)), kinds]
    return F.binary_cross_entropy_with_logits(mine, targets)


def train(model, seed: int = 0, epochs: int = EPOCHS, heard=None, pool_fn=roi_align, prop_fn=proposals) -> list[dict]:
    """Train `model` on the training scenes, as cv-rcnn trains its two-stage detector, with a third loss: for every
    proposal (and true box) that learns a kind, the mask of that kind learns the true mask of its shape, cut to the
    proposal (`target`). pool_fn cuts the boxes for both heads. `heard` hears every 10 batches how far an epoch is,
    with the mean loss of those 10 batches, the three parts together ("loss"), and after every epoch the loss of each
    part and the model."""
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
        runs = {"rpn": 0.0, "second": 0.0, "mask": 0.0}
        reported = 0.0
        for b, i in enumerate(range(0, n, BATCH), 1):
            idx = order[i:i + BATCH]
            maps, rows = model(X[idx].float() / 255)
            first = rpn_loss(rows, C[idx], T[idx])
            rois, classes, ts, mrois, mkinds, mtargets = [], [], [], [], [], []
            for j, s_i in enumerate(idx.tolist()):
                truth, kinds = truths[s_i]
                props = torch.cat([as_boxes(prop_fn(rows[j].detach(), a.clone(), TRAIN_KEEP)), truth])
                ious = torch.as_tensor(iou_matrix(props.numpy(), truth.numpy()))
                best, which = ious.max(dim=1)
                c = torch.where(best >= FG_IOU, kinds[which] + 1, torch.zeros_like(which))
                boxes = props.tolist()
                rois += [(j, tuple(p)) for p in boxes]
                classes.append(c)
                ts.append(offsets(props, truth[which]))
                for q in torch.nonzero(c > 0).flatten().tolist():
                    mrois.append((j, tuple(boxes[q])))
                    mkinds.append(int(kinds[which[q]]))
                    mtargets.append(target(ss[s_i].shapes[int(which[q])].mask, boxes[q]))
            out = model.second(pooled(maps, rois, pool_fn, K).reshape(len(rois), -1))
            second = second_loss(out, torch.cat(classes), torch.cat(ts))
            logits = model.mask_head(pooled(maps, mrois, pool_fn, KM))
            third = mask_loss(logits, torch.tensor(mkinds), torch.stack(mtargets))
            loss = first + second + third
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            runs["rpn"] += first.item()
            runs["second"] += second.item()
            runs["mask"] += third.item()
            if heard is not None and b % 10 == 0:
                heard({"epoch": epoch, "epochs": epochs, "batch": b, "batches": per,
                       "loss": (sum(runs.values()) - reported) / 10})
                reported = sum(runs.values())
        history.append({"epoch": epoch} | {key: v / per for key, v in runs.items()})
        if heard is not None:
            heard({"epoch": epoch, "epochs": epochs} | {key: v / per for key, v in runs.items()} | {"model": model})
    return history


def train_course(seed: int = 0, pool_fn=roi_align, heard=None) -> tuple:
    """The course's Mask R-CNN, trained with `seed` on one thread: (model, history)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = MaskRCNN()
        history = train(model, seed, EPOCHS, heard, pool_fn=pool_fn)
    finally:
        torch.set_num_threads(threads)
    return model, history


def masks_of(model, maps, j: int, found: list, pool_fn=roi_align, paste_fn=paste) -> list:
    """The boxes of one scene (kind, score, box) with the mask the mask head draws for each, of its kind, pasted into
    the scene: (kind, score, box, mask as a bool array)."""
    if not found:
        return []
    logits = model.mask_head(pooled(maps, [(j, tuple(b)) for _, _, b in found], pool_fn, KM))
    probs = torch.sigmoid(logits)
    out = []
    for q, (kd, sc, b) in enumerate(found):
        mask = paste_fn(probs[q, kd].clone(), tuple(b), SIZE)
        out.append((kd, sc, tuple(b), np.asarray(torch.as_tensor(mask).detach().cpu().numpy(), dtype=bool).reshape(SIZE, SIZE)))
    return out


def detect(model, X, keep: int = TEST_KEEP, pool_fn=roi_align, prop_fn=proposals, boxes_fn=boxes_from,
           paste_fn=paste) -> list[tuple]:
    """For every scene of X (bytes): its boxes with their masks (kind, score, box, mask), the highest score first, as
    the two-stage detector finds them, and its proposals; in eval mode, without a graph. The mask head draws a mask
    for every box that is left after NMS, as Mask R-CNN does."""
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
                out = model.second(pooled(maps, [(j, tuple(b)) for b in props.tolist()], pool_fn, K).reshape(len(props), -1))
                found = _TWO.found_boxes(out, props, boxes_fn)
                out_all.append((masks_of(model, maps, j, found, pool_fn, paste_fn), props))
    model.train()
    return out_all


def mask_iou(a, b) -> float:
    """The IoU of two masks: the pixels in both, divided by the pixels in either."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def mask_ap(found_per_scene: list, ss, threshold: float = AP_IOU) -> tuple[float, list[float]]:
    """The mask AP of every kind and their mean: as the mAP of cv-map, with the IoU of the masks instead of the boxes.
    For each kind, the masks of all scenes, the highest score first; a mask finds the shape of its kind with the
    highest mask IoU above the threshold that no mask with a higher score found first."""
    aps = []
    for kind in range(KINDS):
        dets = sorted(((img, sc, mk) for img, f in enumerate(found_per_scene) for k, sc, _, mk in f if k == kind),
                      key=lambda d: -d[1])
        n = sum(1 for s in ss for sh in s.shapes if sh.label == kind)
        taken, tp, total = set(), 0, 0.0
        for r, (img, _, mk) in enumerate(dets):
            best, shape = threshold, None
            for q, sh in enumerate(ss[img].shapes):
                if sh.label == kind and (img, q) not in taken:
                    v = mask_iou(mk, sh.mask)
                    if v > best:
                        best, shape = v, (img, q)
            if shape is not None:
                taken.add(shape)
                tp += 1
                total += tp / (r + 1)
        aps.append(total / n)
    return float(np.mean(aps)), aps


def found_ious(found: list, s) -> tuple[list, int]:
    """For one scene: every shape that a box of 0.5 or more finds, as in cv-rcnn (its kind and a box IoU above 0.5,
    the box with the highest score first, each shape found once), with the IoU of that box's mask and the shape's
    mask: [(kind, mask IoU)]; and how many boxes of 0.5 or more find no shape."""
    taken, out, false = set(), [], 0
    for kd, sc, b, mk in found:
        if sc < SCORE:
            continue
        best, shape = AP_IOU, None
        for q, sh in enumerate(s.shapes):
            if sh.label == kd and q not in taken and iou(b, sh.box) > best:
                best, shape = iou(b, sh.box), q
        if shape is None:
            false += 1
            continue
        taken.add(shape)
        out.append((kd, mask_iou(mk, s.shapes[shape].mask)))
    return out, false


def summary(results: list, ss) -> dict:
    """What the lesson says of a trained Mask R-CNN on the scenes ss: the mask AP at an IoU of 0.5 (and of each kind)
    and of 0.75, the box mAP at 0.5 and 0.75, the shapes found, the false boxes, and for every kind the median IoU of
    the masks of the shapes found."""
    found = [f for f, _ in results]
    boxes = [[(k, sc, b) for k, sc, b, _ in f] for f in found]
    m, aps = mask_ap(found, ss)
    m75, _ = mask_ap(found, ss, 0.75)
    b50, _ = _ONE.mean_ap(boxes, ss)
    b75, _ = _ONE.mean_ap(boxes, ss, 0.75)
    per_kind, false = {k: [] for k in range(KINDS)}, 0
    for f, s in zip(found, ss):
        got, wrong = found_ious(f, s)
        false += wrong
        for kd, v in got:
            per_kind[kd].append(v)
    return {"map": round(m, 4), "aps": [round(a, 4) for a in aps], "map75": round(m75, 4),
            "box_map": round(b50, 4), "box_map75": round(b75, 4),
            "found": sum(len(v) for v in per_kind.values()), "false": false,
            "median": [round(float(np.median(v)), 4) if v else 0.0 for v in per_kind.values()],
            "shapes": sum(len(s.shapes) for s in ss)}


def packed(found: list) -> list[tuple]:
    """The boxes of 0.5 or more with their masks, as the lesson keeps them: (kind, score, box, mask as short text)."""
    return [(k, round(sc, 3), tuple(round(v, 1) for v in b), _FCN.pack(mk)) for k, sc, b, mk in found if sc >= SCORE]


def shown_moment(model, pool_fn=roi_align, prop_fn=proposals, boxes_fn=boxes_from, paste_fn=paste) -> list[list]:
    """The boxes of 0.5 or more and their masks on each of the SHOWN scenes, for the rewind."""
    ss, X = scene_set("test")
    idx = torch.tensor([TEST.index(seed) for seed in SHOWN])
    return [packed(f) for f, _ in detect(model, X[idx], TEST_KEEP, pool_fn, prop_fn, boxes_fn, paste_fn)]


# --- measuring everything again -------------------------------------------------------------------------------------

def touching(ss) -> dict:
    """How many shapes of the scenes ss touch or overlap a shape of the same kind, so that a class map joins them: a
    pixel of one is next to a pixel of the other, also on a diagonal; and in how many scenes that happens."""
    shapes = joined = scenes_with = 0
    for s in ss:
        hit = False
        for i, a in enumerate(s.shapes):
            edge = np.pad(a.mask, 1)
            grown = np.zeros_like(a.mask)
            for dr in (0, 1, 2):
                for dc in (0, 1, 2):
                    grown |= edge[dr:dr + SIZE, dc:dc + SIZE]
            t = any(j != i and b.label == a.label and (grown & b.mask).any() for j, b in enumerate(s.shapes))
            shapes += 1
            joined += t
            hit |= t
        scenes_with += hit
    return {"shapes": shapes, "joined": joined, "scenes": scenes_with}


def box_share(ss) -> list[float]:
    """For every kind, the median share of the pixels of a shape's box that are the shape: what a box would score as
    a mask."""
    shares = {k: [] for k in range(KINDS)}
    for s in ss:
        for sh in s.shapes:
            t, l, b, r = sh.box
            shares[sh.label].append(int(sh.mask.sum()) / ((b - t + 1) * (r - l + 1)))
    return [round(float(np.median(v)), 4) for v in shares.values()]


def round_trip(ss, m: int) -> float:
    """The median IoU of the true masks of the shapes of ss with themselves after `target` in their own box with m by
    m bins and `paste` back: how much a mask of m by m can hold."""
    out = []
    for s in ss:
        for sh in s.shapes:
            out.append(mask_iou(paste(target(sh.mask, sh.box, m), sh.box).numpy(), sh.mask))
    return round(float(np.median(out)), 4)


def measure_seed(seed: int, pool_fn=roi_align, model_file: str | None = None) -> dict:
    """Train with `seed` and measure everything TRAINED holds for one seed; with model_file, the trained model is
    saved there too, for the pictures."""
    model, history = train_course(seed, pool_fn)
    if model_file:
        torch.save(model.state_dict(), model_file)
    ss, Xt = scene_set("test")
    out = summary(detect(model, Xt, TEST_KEEP, pool_fn), ss)
    out["curves"] = {key: [round(h[key], 4) for h in history] for key in ("rpn", "second", "mask")}
    return out


def pictures_of(model, pool_fn=roi_align) -> dict:
    """What the pictures need of the course's trained Mask R-CNN (seed 0): its boxes and masks on the scene of the
    worked example, and its mask AP on the test scenes at every threshold of the slider."""
    s = example()
    found, _ = detect(model, torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0), TEST_KEEP, pool_fn)[0]
    ss, Xt = scene_set("test")
    masks = [f for f, _ in detect(model, Xt, TEST_KEEP, pool_fn)]
    return {"scene": packed(found), "thresholds": {t: round(mask_ap(masks, ss, t)[0], 4) for t in THRESHOLDS}}


# --- exercise 1: bilinear ---------------------------------------------------------------------------------------------

_DOTS = _ONE._DOTS
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close
_kept = _FCN._kept

_B_HEAD = ("import torch\n\n\ndef bilinear(maps, ys, xs):\n"
           "    # maps: shape (C, H, W); ys and xs: lists of positions in squares, square s has its middle at s\n"
           "    # give back the bilinear value at every (y, x) of the grid ys by xs: shape (C, len(ys), len(xs))\n"
           "    h, w = maps.shape[1], maps.shape[2]\n"
           "    ys = [min(max(y, 0), h - 1) for y in ys]   # a point outside the maps takes the value at the edge\n"
           "    xs = [min(max(x, 0), w - 1) for x in xs]\n")
_B_START = _B_HEAD + "    ...\n"
_B_ROWS = "    r0 = [int(y) for y in ys]\n    r1 = [min(r + 1, h - 1) for r in r0]\n"
_B_COLS = "    c0 = [int(x) for x in xs]\n    c1 = [min(c + 1, w - 1) for c in c0]\n"
_B_T = "    t = (torch.tensor(ys) - torch.tensor(r0)).reshape(-1, 1)   # a column: one share for every row\n"
_B_U = "    u = torch.tensor(xs) - torch.tensor(c0)\n"
_B_TOP = "    top = (1 - u) * maps[:, r0][:, :, c0] + u * maps[:, r0][:, :, c1]\n"
_B_BOTTOM = "    bottom = (1 - u) * maps[:, r1][:, :, c0] + u * maps[:, r1][:, :, c1]\n"
_B_RETURN = "    return (1 - t) * top + t * bottom\n"


def _b(rows=_B_ROWS, cols=_B_COLS, t=_B_T, u=_B_U, top=_B_TOP, bottom=_B_BOTTOM, ret=_B_RETURN) -> str:
    return _B_HEAD + rows + cols + t + u + top + bottom + ret


_B = _b()
_B_TWO = "import torch\n\nmaps = torch.tensor([[[0.0, 100.0], [200.0, 40.0]]])\n"
_B_SCENARIOS = [
    # the 2 by 2 pixels of value_at in cv-resize-interpolation, as maps of one channel, and one point between them
    _B_TWO + "bilinear(maps, [0.25], [0.5]).tolist()",
    # 3 rows and 2 columns of points
    _B_TWO + "bilinear(maps, [0, 0.25, 1], [0, 0.5]).tolist()",
    # points outside the maps
    _B_TWO + "bilinear(maps, [-1.0, 1.5], [0.25, 3.0]).tolist()",
    # 2 channels of 3 by 4
    "import torch\n\nmaps = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(0))\n"
    "bilinear(maps, [0.5, 1.75], [0.25, 2.5, 3.0]).tolist()",
    # training follows the values back to the maps: how much each square counts for one point
    "import torch\n\nmaps = torch.ones(1, 2, 2, requires_grad=True)\nbilinear(maps, [0.25], [0.5]).sum().backward()\n"
    "maps.grad.tolist()",
    # 64 maps of 16 by 16, as the backbone gives them
    "import torch\n\nmaps = torch.randn(64, 16, 16, generator=torch.Generator().manual_seed(1))\n"
    "out = bilinear(maps, [4.75, 5.75, 6.75, 7.75], [4.375, 5.375, 6.375, 7.375])\n[list(out.shape), out[0].tolist()]",
]
_PAIRS_TOP = "    top = (1 - u) * maps[:, r0, c0] + u * maps[:, r0, c1]\n"
_PAIRS_BOTTOM = "    bottom = (1 - u) * maps[:, r1, c0] + u * maps[:, r1, c1]\n"
_B_WRONG = {
    "pairs": _b(top=_PAIRS_TOP, bottom=_PAIRS_BOTTOM),
    "weights": _b(top="    top = u * maps[:, r0][:, :, c0] + (1 - u) * maps[:, r0][:, :, c1]\n",
                  bottom="    bottom = u * maps[:, r1][:, :, c0] + (1 - u) * maps[:, r1][:, :, c1]\n",
                  ret="    return t * top + (1 - t) * bottom\n"),
}


def _no_grad(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "does not require grad" in str(e)


def _past_edge(e: BaseException) -> bool:
    return isinstance(e, IndexError) and "out of bounds" in str(e)


def _broadcast(e: BaseException) -> bool:
    return isinstance(e, (IndexError, RuntimeError)) and re.search(r"could not be broadcast|must match the size", str(e)) is not None


_B_MISTAKES = [
    Mistake("edge", error=_past_edge,
            en="Python stops when your code takes squares out of `maps`: a square past the last one. A point on the last row has no row under it, so the row under a point is at most the last: `r1 = [min(r + 1, h - 1) for r in r0]`, and the same for the columns with `w - 1`.",
            nl="Python stopt als je code vakjes uit `maps` haalt: een vakje voorbij het laatste. Een punt op de laatste rij heeft geen rij eronder, dus de rij onder een punt is hoogstens de laatste: `r1 = [min(r + 1, h - 1) for r in r0]`, en hetzelfde voor de kolommen met `w - 1`."),
    Mistake("grad", error=_no_grad,
            en="Training cannot follow your values back to the maps: they are no longer made from `maps` itself, for example because of `.item()`, `float(...)` or `.tolist()` on numbers of the maps, or a new `torch.tensor` of them. Mix the squares of `maps` itself: `maps[:, r0][:, :, c0]` and so on.",
            nl="Training kan je waarden niet terugvolgen naar de maps: ze zijn niet meer van `maps` zelf gemaakt, bijvoorbeeld door `.item()`, `float(...)` of `.tolist()` op getallen van de maps, of een nieuwe `torch.tensor` ervan. Meng de vakjes van `maps` zelf: `maps[:, r0][:, :, c0]` enzovoort."),
    Mistake("pairs", wrong=_B_WRONG["pairs"],
            en=_S_EN + ": `maps[:, r0, c0]` takes one square for every pair of a row and a column, the first row with the first column, and so on, not every row with every column. Take the rows first and then the columns: `maps[:, r0][:, :, c0]` has shape (C, len(ys), len(xs)).",
            nl=_S_NL + ": `maps[:, r0, c0]` neemt één vakje voor elk paar van een rij en een kolom, de eerste rij met de eerste kolom, enzovoort, niet elke rij met elke kolom. Neem eerst de rijen en dan de kolommen: `maps[:, r0][:, :, c0]` heeft shape (C, len(ys), len(xs))."),
    Mistake("shapes", error=_broadcast,
            en="Python stops when it combines your numbers: their shapes do not fit. Does every share of a row multiply a whole row of the grid? `t` must be a column, `.reshape(-1, 1)`, and `u` a row. And take every row with every column: `maps[:, r0][:, :, c0]` has shape (C, len(ys), len(xs)), where `maps[:, r0, c0]` takes one square for every pair.",
            nl="Python stopt als hij je getallen combineert: hun shapes passen niet. Vermenigvuldigt elk deel van een rij een hele rij van het raster? `t` moet een kolom zijn, `.reshape(-1, 1)`, en `u` een rij. En neem elke rij met elke kolom: `maps[:, r0][:, :, c0]` heeft shape (C, len(ys), len(xs)), waar `maps[:, r0, c0]` één vakje voor elk paar neemt."),
    Mistake("weights", wrong=_B_WRONG["weights"],
            en=_S_EN + ": your weights are the wrong way round: the square farther away counts more. The nearer square counts more: `(1 - u)` goes with column `c0`, `u` with `c1`, and `(1 - t)` with row `r0`, `t` with `r1`.",
            nl=_S_NL + ": je weights staan verkeerd om: het vakje dat verder weg ligt, telt zwaarder. Het dichtere vakje telt zwaarder: `(1 - u)` hoort bij kolom `c0`, `u` bij `c1`, en `(1 - t)` bij rij `r0`, `t` bij `r1`."),
]

# --- exercise 2: roi_align --------------------------------------------------------------------------------------------

_A_HEAD = ("from cnn.bilinear import bilinear\n\n\ndef roi_align(maps, box, k):\n"
           "    # maps: the feature maps of one scene, shape (C, H, W), every square 8 by 8 pixels\n"
           "    # box: (top, left, bottom, right) in pixels, both ends included; it may stick out of the scene\n"
           "    # give back the bilinear value at the middle of each of k by k bins of the box: (C, k, k)\n"
           "    top, left, bottom, right = box\n")
_A_START = _A_HEAD + "    ...\n"
_A_SIZE = "    h, w = bottom - top + 1, right - left + 1\n"
_A_YS = "    ys = [top - 0.5 + (i + 0.5) * h / k for i in range(k)]   # the middles of the bins, in pixels\n"
_A_XS = "    xs = [left - 0.5 + (j + 0.5) * w / k for j in range(k)]\n"
_A_RETURN = "    return bilinear(maps, [(y - 3.5) / 8 for y in ys], [(x - 3.5) / 8 for x in xs])\n"


def _a(size=_A_SIZE, ys=_A_YS, xs=_A_XS, ret=_A_RETURN) -> str:
    return _A_HEAD + size + ys + xs + ret


_A = _a()
_A_MAPS = "import torch\n\nmaps = torch.randperm({n}, generator=torch.Generator().manual_seed({seed})).reshape({shape}).float()\n"
_A_PHOTO = ("import torch\nfrom everything_ai.scenes import scene\n\ns = scene(18)\n"
            "photo = torch.tensor(s.image).permute(2, 0, 1).float() / 255\n"
            "maps = photo.reshape(3, 16, 8, 16, 8).mean(dim=(2, 4))   # every square the mean color of its pixels\n")
_A_SCENARIOS = [
    # 6 by 6 squares, as in roi_pool of cv-rcnn: a box over rows 0 to 3 and columns 1 to 5, 2 by 2 bins
    _A_MAPS.format(n=2 * 36, seed=0, shape="2, 6, 6") + "roi_align(maps, (5.5, 9.0, 30.2, 44.0), 2).tolist()",
    # a box that sticks out of the scene, above and to the right
    _A_MAPS.format(n=36, seed=1, shape="1, 6, 6") + "roi_align(maps, (-6.0, 30.0, 20.0, 60.0), 2).tolist()",
    # a box inside one square: RoI pooling gave every bin that square
    _A_MAPS.format(n=36, seed=2, shape="1, 6, 6") + "roi_align(maps, (17.0, 17.5, 22.0, 23.0), 2).tolist()",
    # the maps of a scene of 128 by 128 pixels, the circle of the worked example, 4 by 4 bins
    _A_MAPS.format(n=3 * 256, seed=3, shape="3, 16, 16") + "roi_align(maps, (58, 50, 86, 78), 4)[1].tolist()",
    # maps of another size, and a box below them
    _A_MAPS.format(n=3 * 12 * 10, seed=4, shape="3, 12, 10") + "roi_align(maps, (100.5, 70.0, 130.0, 90.0), 2).tolist()",
    # the scene of the worked example, every square its mean color: the circle in 16 by 16 bins
    _A_PHOTO + "out = roi_align(maps, s.shapes[3].box, 16)\n[list(out.shape), out[:, 8, 8].tolist()]",
]
_A_WRONG = {
    "rounded": _a(ret="    return bilinear(maps, [(y - 3.5) // 8 for y in ys], [(x - 3.5) // 8 for x in xs])\n"),
    "no-middle": _a(ret="    return bilinear(maps, [y / 8 for y in ys], [x / 8 for x in xs])\n"),
    "edges": _a(ys="    ys = [top - 0.5 + i * h / k for i in range(k)]\n", xs="    xs = [left - 0.5 + j * w / k for j in range(k)]\n"),
    "plus-one": _a(size="    h, w = bottom - top, right - left\n", ys="    ys = [top + (i + 0.5) * h / k for i in range(k)]\n",
                   xs="    xs = [left + (j + 0.5) * w / k for j in range(k)]\n"),
    "sixteen": _a(ret="    return bilinear(maps, [(y - 7.5) / 16 for y in ys], [(x - 7.5) / 16 for x in xs])\n"),
    "swapped": _a(ret="    return bilinear(maps, [(x - 3.5) / 8 for x in xs], [(y - 3.5) / 8 for y in ys])\n"),
    "pixels": _a(ret="    return bilinear(maps, ys, xs)\n"),
}
_A_MISTAKES = [
    Mistake("rounded", wrong=_A_WRONG["rounded"],
            en=_S_EN + ": `//` rounds every middle down to a whole square, as RoI pooling did. RoIAlign keeps where in the square a middle lies: `(y - 3.5) / 8`, and `bilinear` mixes the squares around it.",
            nl=_S_NL + ": `//` rondt elk midden af naar beneden, op een heel vakje, zoals RoI pooling deed. RoIAlign houdt waar in het vakje een midden ligt: `(y - 3.5) / 8`, en `bilinear` mengt de vakjes eromheen."),
    Mistake("no-middle", wrong=_A_WRONG["no-middle"],
            en=_S_EN + ": your places are 3.5 pixels, almost half a square, off. Square s holds the pixels 8s to 8s + 7, so its middle is pixel 8s + 3.5, and `bilinear` counts from the middle of square 0: pixel y lies at `(y - 3.5) / 8`.",
            nl=_S_NL + ": je plaatsen liggen 3,5 pixel, bijna een half vakje, te ver. Vakje s bevat de pixels 8s tot en met 8s + 7, dus zijn midden is pixel 8s + 3,5, en `bilinear` telt vanaf het midden van vakje 0: pixel y ligt op `(y - 3.5) / 8`."),
    Mistake("edges", wrong=_A_WRONG["edges"],
            en=_S_EN + ": your points lie at the start of every bin. RoIAlign takes the middle of every bin: bin i starts `i * h / k` pixels into the box, and its middle lies half a bin further, `(i + 0.5) * h / k`.",
            nl=_S_NL + ": je punten liggen aan het begin van elke bin. RoIAlign neemt het midden van elke bin: bin i begint `i * h / k` pixels in de box, en zijn midden ligt een halve bin verder, `(i + 0.5) * h / k`."),
    Mistake("plus-one", wrong=_A_WRONG["plus-one"],
            en=_S_EN + ": your box runs from the middle of its top pixel to the middle of its bottom pixel. Both ends are in the box, as everywhere in the course, so it is `bottom - top + 1` pixels high, from the top edge of pixel `top`, `top - 0.5`, to the bottom edge of pixel `bottom`.",
            nl=_S_NL + ": je box loopt van het midden van zijn bovenste pixel tot het midden van zijn onderste pixel. Beide einden horen bij de box, zoals overal in de course, dus hij is `bottom - top + 1` pixels hoog, van de bovenrand van pixel `top`, `top - 0.5`, tot de onderrand van pixel `bottom`."),
    Mistake("sixteen", wrong=_A_WRONG["sixteen"],
            en=_S_EN + ": you divide by 16, the stride of the maps of the paper's ResNet. Here every square stands for 8 by 8 pixels: `(y - 3.5) / 8`.",
            nl=_S_NL + ": je deelt door 16, de stride van de maps van de ResNet in het paper. Hier staat elk vakje voor 8 bij 8 pixels: `(y - 3.5) / 8`."),
    Mistake("swapped", wrong=_A_WRONG["swapped"],
            en=_S_EN + ": the rows and the columns are swapped. `bilinear` takes the rows first, from `top` and `bottom`, then the columns, from `left` and `right`.",
            nl=_S_NL + ": de rijen en de kolommen zijn verwisseld. `bilinear` neemt eerst de rijen, uit `top` en `bottom`, en dan de kolommen, uit `left` en `right`."),
    Mistake("pixels", wrong=_A_WRONG["pixels"],
            en=_S_EN + ": you give `bilinear` places in pixels, and it counts in squares of 8 by 8 pixels. Pixel y lies at `(y - 3.5) / 8` squares from the middle of square 0.",
            nl=_S_NL + ": je geeft `bilinear` plaatsen in pixels, en hij telt in vakjes van 8 bij 8 pixels. Pixel y ligt `(y - 3.5) / 8` vakjes van het midden van vakje 0."),
]

# --- exercise 3: paste ------------------------------------------------------------------------------------------------

_P_HEAD = ("import math\nimport torch\nfrom cnn.bilinear import bilinear\n\n\ndef paste(mask, box, size):\n"
           "    # mask: the probabilities the mask head gives one box, shape (m, m)\n"
           "    # box: (top, left, bottom, right) in pixels, both ends included; size: the scene is size by size pixels\n"
           "    # give back the mask in the scene, shape (size, size): True where the mask, made as large as the box,\n"
           "    # is 0.5 or more\n"
           "    top, left, bottom, right = box\n    m = mask.shape[0]\n"
           "    out = torch.zeros(size, size, dtype=torch.bool)\n")
_P_START = _P_HEAD + "    ...\n    return out\n"
_P_ROWS = ("    # the pixels whose middle lies in the box, and in the scene\n"
           "    r0, r1 = max(math.ceil(top - 0.5), 0), min(math.floor(bottom + 0.5), size - 1)\n")
_P_COLS = "    c0, c1 = max(math.ceil(left - 0.5), 0), min(math.floor(right + 0.5), size - 1)\n"
_P_YS = ("    # where their middles lie in the mask, counted from the middle of its first row and column\n"
         "    ys = [(r - top + 0.5) / (bottom - top + 1) * m - 0.5 for r in range(r0, r1 + 1)]\n")
_P_XS = "    xs = [(c - left + 0.5) / (right - left + 1) * m - 0.5 for c in range(c0, c1 + 1)]\n"
_P_PUT = "    out[r0:r1 + 1, c0:c1 + 1] = bilinear(mask.unsqueeze(0), ys, xs)[0] >= 0.5\n"
_P_RETURN = "    return out\n"


def _p(rows=_P_ROWS, cols=_P_COLS, ys=_P_YS, xs=_P_XS, put=_P_PUT, ret=_P_RETURN) -> str:
    return _P_HEAD + rows + cols + ys + xs + put + ret


_P = _p()
_P_FOUR = ("import torch\n\nmask = torch.tensor([[0.1, 0.5, 0.6, 0.1], [0.6, 0.9, 0.9, 0.6],\n"
           "                     [0.6, 0.9, 0.9, 0.6], [0.1, 0.6, 0.6, 0.1]])\n")
_P_SIXTEEN = ("import torch\n\nmask = torch.zeros(16, 16)\nmask[4:12, 2:14] = 0.9   # a band across the middle of the box\n")
_P_ROUND = ("import torch\n\n# the mask head's probabilities for a circle: 1 at the middle, 0 at the edge\n"
            "middle = [(i - 7.5) ** 2 for i in range(16)]\n"
            "mask = torch.tensor([[1 - (a + b) / 64 for b in middle] for a in middle])\n")
_P_SCENARIOS = [
    # a mask of 4 by 4 in a box of 6 by 4 pixels, in a scene of 8 by 8
    _P_FOUR + "paste(mask, (1.0, 2.0, 6.0, 5.0), 8).int().tolist()",
    # a box that sticks out of the scene, above and to the right
    _P_FOUR + "paste(mask, (-2.0, 4.0, 4.0, 10.0), 8).int().tolist()",
    # a box that does not start at a whole pixel
    _P_FOUR + "paste(mask, (1.4, 0.6, 5.7, 6.2), 8).int().tolist()",
    # a mask of 16 by 16 in a box of 32 by 32, in a scene of 128 by 128
    _P_SIXTEEN + "out = paste(mask, (40, 30, 71, 61), 128)\n[int(out.sum()), int(out[:, 45].sum()), int(out[56].sum())]",
    # a round mask in a box of a scene of 128 by 128
    _P_ROUND + "out = paste(mask, (30.6, 20.2, 89.4, 109.8), 128)\n[int(out.sum()), int(out[60].sum()), int(out[:, 65].sum())]",
]
_P_WRONG = {
    "no-clip": _p(rows="    r0, r1 = math.ceil(top - 0.5), math.floor(bottom + 0.5)\n",
                  cols="    c0, c1 = math.ceil(left - 0.5), math.floor(right + 0.5)\n"),
    "plus-one": _p(ys="    ys = [(r - top) / (bottom - top) * m - 0.5 for r in range(r0, r1 + 1)]\n",
                   xs="    xs = [(c - left) / (right - left) * m - 0.5 for c in range(c0, c1 + 1)]\n"),
    "no-minus-half": _p(ys="    ys = [(r - top + 0.5) / (bottom - top + 1) * m for r in range(r0, r1 + 1)]\n",
                        xs="    xs = [(c - left + 0.5) / (right - left + 1) * m for c in range(c0, c1 + 1)]\n"),
    "corner": _p(ys="", xs="", put="    out[r0:r0 + m, c0:c0 + m] = mask >= 0.5\n"),
    "swapped": _p(put="    out[r0:r1 + 1, c0:c1 + 1] = bilinear(mask.unsqueeze(0), xs, ys)[0].T >= 0.5\n"),
}


def _value_shape(e: BaseException) -> bool:
    return isinstance(e, (RuntimeError, IndexError)) and re.search(r"shape mismatch|must match the size|could not be broadcast|"
                                                                   r"expanded size", str(e)) is not None


_P_MISTAKES = [
    Mistake("not-whole", error=lambda e: isinstance(e, TypeError) and re.search(r"slice indices must be integers|integer", str(e)) is not None,
            en="Python stops when your code counts the pixels of the box with `range`, or takes them out of `out`: a pixel must be a whole number, and a box of floats gives floats, such as `40.6`. `math.ceil` and `math.floor` give whole numbers: the first pixel whose middle lies in the box is `math.ceil(top - 0.5)`, the last `math.floor(bottom + 0.5)`.",
            nl="Python stopt als je code de pixels van de box met `range` telt, of uit `out` haalt: een pixel moet een geheel getal zijn, en een box van floats geeft floats, zoals `40.6`. `math.ceil` en `math.floor` geven gehele getallen: de eerste pixel waarvan het midden in de box ligt, is `math.ceil(top - 0.5)`, de laatste `math.floor(bottom + 0.5)`."),
    Mistake("no-clip", error=_value_shape,
            en="Python stops when your code puts the mask into `out`: the pixels of the box and the part of `out` you give them have other shapes. A box may stick out of the scene; keep only its pixels in the scene, from 0 to `size - 1`: `max(math.ceil(top - 0.5), 0)` and `min(math.floor(bottom + 0.5), size - 1)`, and the same for the columns. And make `ys` and `xs` from those same rows and columns.",
            nl="Python stopt als je code het mask in `out` zet: de pixels van de box en het deel van `out` dat je ze geeft, hebben verschillende shapes. Een box kan uit de scène steken; houd alleen zijn pixels in de scène, van 0 tot en met `size - 1`: `max(math.ceil(top - 0.5), 0)` en `min(math.floor(bottom + 0.5), size - 1)`, en hetzelfde voor de kolommen. En maak `ys` en `xs` van diezelfde rijen en kolommen."),
    Mistake("plus-one", wrong=_P_WRONG["plus-one"],
            en=_S_EN + ": your box runs from the middle of its top pixel to the middle of its bottom pixel. Both ends are in the box, so it is `bottom - top + 1` pixels high, and the middle of pixel r lies `r - top + 0.5` pixels below its top edge: a share `(r - top + 0.5) / (bottom - top + 1)` of the box.",
            nl=_S_NL + ": je box loopt van het midden van zijn bovenste pixel tot het midden van zijn onderste pixel. Beide einden horen bij de box, dus hij is `bottom - top + 1` pixels hoog, en het midden van pixel r ligt `r - top + 0.5` pixels onder zijn bovenrand: een deel `(r - top + 0.5) / (bottom - top + 1)` van de box."),
    Mistake("no-minus-half", wrong=_P_WRONG["no-minus-half"],
            en=_S_EN + ": your places in the mask are half a row off. `bilinear` counts from the middle of row 0 of the mask, and a share of the box times `m` counts from its top edge: subtract 0.5, `... * m - 0.5`.",
            nl=_S_NL + ": je plaatsen in het mask liggen een halve rij te ver. `bilinear` telt vanaf het midden van rij 0 van het mask, en een deel van de box keer `m` telt vanaf zijn bovenrand: trek er 0,5 af, `... * m - 0.5`."),
    Mistake("corner", wrong=_P_WRONG["corner"],
            en=_S_EN + ": you put the mask in the scene as it is, m by m pixels from the corner of the box. It must first be made as large as the box: for every pixel of the box, where its middle lies in the mask, and the bilinear value there.",
            nl=_S_NL + ": je zet het mask zoals het is in de scène, m bij m pixels vanaf de hoek van de box. Het moet eerst zo groot worden als de box: voor elke pixel van de box, waar zijn midden in het mask ligt, en de bilineaire waarde daar."),
    Mistake("swapped", wrong=_P_WRONG["swapped"],
            en=_S_EN + ": the rows and the columns are swapped. `bilinear` takes the rows of the mask first, `ys`, from `top` and `bottom`, then its columns, `xs`, from `left` and `right`.",
            nl=_S_NL + ": de rijen en de kolommen zijn verwisseld. `bilinear` neemt eerst de rijen van het mask, `ys`, uit `top` en `bottom`, en dan de kolommen, `xs`, uit `left` en `right`."),
]


# --- the exercises ----------------------------------------------------------------------------------------------------

def _yours():
    from everything_ai import yours
    return yours


EXERCISES = {
    "bilinear": ScenarioBox(
        name="bilinear",
        start=_B_START,
        solution=_B,
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "bilinear"),
        picture=_yours().arrays(call=1, result="bilinear(maps, [0, 0.25, 1], [0, 0.5])[0]", shade=True),
        first=[_DOTS],
        mistakes=_B_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the squares above and under every point and left and right of it, how far the point lies from the first of each, and the mix of the four squares, for every point of the grid at once.",
                      "nl": "Vervang de `...` door je code: de vakjes boven en onder elk punt en links en rechts ervan, hoe ver het punt van de eerste van elk ligt, en de mix van de vier vakjes, voor elk punt van het raster tegelijk."},
            "wrong": {"en": _S_EN + ". Do you mix along the rows first with `u`, from `xs`, and then between them with `t`, from `ys`, the nearer square counting more, as in your `value_at`?",
                      "nl": _S_NL + ". Meng je eerst langs de rijen met `u`, uit `xs`, en dan tussen ze in met `t`, uit `ys`, waarbij het dichtere vakje zwaarder telt, zoals in je `value_at`?"},
            "almost": {"en": "Right for one point. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of rows and columns, for points outside the maps, and for maps of several channels?",
                       "nl": "Goed voor één punt. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal rijen en kolommen, voor punten buiten de maps, en voor maps met meer channels?"},
            "correct": {"en": "Your `bilinear` is right: the value of the maps at every point of the grid, for every channel at once, and training can follow it back to the maps. It is now in your project, as `{saved}`.",
                        "nl": "Je `bilinear` klopt: de waarde van de maps op elk punt van het raster, voor elk channel tegelijk, en training kan hem terugvolgen naar de maps. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("bilinear"),
        },
    ),
    "roi-align": ScenarioBox(
        name="roi-align",
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "roi_align"),
        picture=_yours().image(call=5, result="roi_align(maps, s.shapes[3].box, 16).permute(1, 2, 0)", size=128),
        first=[_DOTS],
        mistakes=_A_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the middles of the k bins down and across the box, in pixels, then in squares, and `bilinear` at them.",
                      "nl": "Vervang de `...` door je code: de middens van de k bins omlaag en opzij in de box, in pixels, dan in vakjes, en `bilinear` daarop."},
            "wrong": {"en": _S_EN + ". Is the box `bottom - top + 1` pixels high, from `top - 0.5`, does bin i have its middle `(i + 0.5) * h / k` into it, and does pixel y lie at `(y - 3.5) / 8` squares?",
                      "nl": _S_NL + ". Is de box `bottom - top + 1` pixels hoog, vanaf `top - 0.5`, ligt het midden van bin i `(i + 0.5) * h / k` erin, en ligt pixel y op `(y - 3.5) / 8` vakjes?"},
            "almost": {"en": "Right for the first box. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for maps of any size, any k, and a box that sticks out?",
                       "nl": "Goed voor de eerste box. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor maps van elke grootte, elke k, en een box die uitsteekt?"},
            "correct": {"en": "Your `roi_align` is right: every box, of any size, becomes k by k numbers per channel, taken where its bins really lie, without rounding. It is now in your project, as `{saved}`.",
                        "nl": "Je `roi_align` klopt: elke box, van elke grootte, wordt k bij k getallen per channel, genomen waar zijn bins echt liggen, zonder af te ronden. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("roi_align"),
        },
    ),
    "paste": ScenarioBox(
        name="paste",
        start=_P_START,
        solution=_P,
        scenarios=_P_SCENARIOS,
        compare=_close,
        save_as=("cnn", "paste"),
        picture=_yours().image(call=4, result="paste(mask, (30.6, 20.2, 89.4, 109.8), 128)", size=128),
        first=[_DOTS],
        mistakes=_P_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the rows and columns of the pixels in the box and the scene, where their middles lie in the mask, and `bilinear` of the mask there, 0.5 or more, into `out`.",
                      "nl": "Vervang de `...` door je code: de rijen en kolommen van de pixels in de box en de scène, waar hun middens in het mask liggen, en `bilinear` van het mask daar, 0,5 of meer, in `out`."},
            "wrong": {"en": _S_EN + ". Do you take the pixels from `math.ceil(top - 0.5)` to `math.floor(bottom + 0.5)`, kept in the scene, and is the place of pixel r in the mask `(r - top + 0.5) / (bottom - top + 1) * m - 0.5`?",
                      "nl": _S_NL + ". Neem je de pixels van `math.ceil(top - 0.5)` tot en met `math.floor(bottom + 0.5)`, binnen de scène, en is de plaats van pixel r in het mask `(r - top + 0.5) / (bottom - top + 1) * m - 0.5`?"},
            "almost": {"en": "Right for the first box. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for a box that sticks out of the scene, one that does not start at a whole pixel, and masks of any size?",
                       "nl": "Goed voor de eerste box. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor een box die uit de scène steekt, een box die niet bij een hele pixel begint, en masks van elke grootte?"},
            "correct": {"en": "Your `paste` is right: the small mask of a box, made as large as the box and cut off at 0.5, in its place in the scene. It is now in your project, as `{saved}`.",
                        "nl": "Je `paste` klopt: het kleine mask van een box, zo groot gemaakt als de box en afgekapt bij 0,5, op zijn plaats in de scène. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("paste"),
        },
    ),
}

# The lesson is done when these are correct. "learns" is a choice, "why-align" the open question.
REQUIRED = ["bilinear", "roi-align", "paste", "learns", "why-align"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bilinear", _B_START, "empty", None),
    ("bilinear", _B, "correct", None),
    # every point on its own, as value_at, into a tensor of zeros: training follows that back too
    ("bilinear", _B_HEAD + "    out = torch.zeros(maps.shape[0], len(ys), len(xs))\n"
                 "    for i in range(len(ys)):\n        for j in range(len(xs)):\n"
                 "            r0, c0 = int(ys[i]), int(xs[j])\n            r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)\n"
                 "            t, u = ys[i] - r0, xs[j] - c0\n"
                 "            top = (1 - u) * maps[:, r0, c0] + u * maps[:, r0, c1]\n"
                 "            bottom = (1 - u) * maps[:, r1, c0] + u * maps[:, r1, c1]\n"
                 "            out[:, i, j] = (1 - t) * top + t * bottom\n    return out\n", "correct", None),
    # the shares as lists first
    ("bilinear", _b(t="    t = torch.tensor([ys[i] - r0[i] for i in range(len(ys))]).reshape(-1, 1)\n",
                    u="    u = torch.tensor([xs[j] - c0[j] for j in range(len(xs))])\n"), "correct", None),
    # every point on its own, from the numbers of the maps as a list: right, but training cannot follow it back
    ("bilinear", _B_HEAD + "    grid = maps.tolist()\n    out = []\n    for chan in grid:\n        rows = []\n"
                 "        for y in ys:\n            row = []\n            for x in xs:\n"
                 "                r0, c0 = int(y), int(x)\n                r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)\n"
                 "                t, u = y - r0, x - c0\n"
                 "                top = (1 - u) * chan[r0][c0] + u * chan[r0][c1]\n"
                 "                bottom = (1 - u) * chan[r1][c0] + u * chan[r1][c1]\n"
                 "                row.append((1 - t) * top + t * bottom)\n            rows.append(row)\n"
                 "        out.append(rows)\n    return torch.tensor(out)\n", "error", "grad"),
    ("bilinear", _b(rows="    r0 = [int(y) for y in ys]\n    r1 = [r + 1 for r in r0]\n"), "error", "edge"),
    ("bilinear", _b(t="    t = torch.tensor(ys) - torch.tensor(r0)\n"), "error", "shapes"),
    # t from xs and u from ys: for one point the numbers come out, but mixed the wrong way round
    ("bilinear", _b(t="    t = (torch.tensor(xs) - torch.tensor(c0)).reshape(-1, 1)\n", u="    u = torch.tensor(ys) - torch.tensor(r0)\n"),
     "wrong", None),
    ("bilinear", _b(ret="    return ...\n"), "wrong", "dots-left"),
] + [("bilinear", code, "wrong", key) for key, code in _B_WRONG.items()] + [
    ("roi-align", _A_START, "empty", None),
    ("roi-align", _A, "correct", None),
    # in one step from pixels to squares: (top - 0.5 + ... - 3.5) / 8 is (top - 4 + ...) / 8
    ("roi-align", _A_HEAD + "    h, w = bottom - top + 1, right - left + 1\n"
                  "    ys = [(top - 4 + (i + 0.5) * h / k) / 8 for i in range(k)]\n"
                  "    xs = [(left - 4 + (j + 0.5) * w / k) / 8 for j in range(k)]\n    return bilinear(maps, ys, xs)\n", "correct", None),
    ("roi-align", _a(ret="    return ...\n"), "wrong", "dots-left"),
    # 16 as in the paper, with the middle of a square of 8
    ("roi-align", _a(ret="    return bilinear(maps, [(y - 3.5) / 16 for y in ys], [(x - 3.5) / 16 for x in xs])\n"), "wrong", None),
] + [("roi-align", code, "wrong", key) for key, code in _A_WRONG.items()] + [
    ("paste", _P_START, "empty", None),
    ("paste", _P, "correct", None),
    # every pixel of the box on its own, with bilinear for one point at a time
    ("paste", _P_HEAD + _P_ROWS + _P_COLS + "    for r in range(r0, r1 + 1):\n        for c in range(c0, c1 + 1):\n"
              "            y = (r - top + 0.5) / (bottom - top + 1) * m - 0.5\n"
              "            x = (c - left + 0.5) / (right - left + 1) * m - 0.5\n"
              "            out[r, c] = bilinear(mask.unsqueeze(0), [y], [x])[0, 0, 0] >= 0.5\n" + _P_RETURN, "correct", None),
    ("paste", _p(rows="    r0, r1 = max(top - 0.5, 0), min(bottom + 0.5, size - 1)\n"), "error", "not-whole"),
    ("paste", _P_WRONG["no-clip"], "error", "no-clip"),
    # above 0.5 instead of 0.5 or more: the first case has a pixel of exactly 0.5
    ("paste", _p(put=_P_PUT.replace(">= 0.5", "> 0.5")), "wrong", None),
    ("paste", _p(put="    ...\n"), "wrong", "dots-left"),
] + [("paste", code, "wrong", key) for key, code in _P_WRONG.items() if key != "no-clip"]


# --- the worked example ---------------------------------------------------------------------------------------------

def worked() -> dict:
    """The worked example's numbers, as its cell computes them: the backbone of cv-rcnn and the mask head, untrained,
    built after torch.manual_seed(0); RoIAlign of the circle's box by hand, its target, and the loss of its mask."""
    s = example()
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    )
    mask_head = nn.Sequential(
        nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 2, stride=2), nn.ReLU(),
        nn.Conv2d(32, 3, 1),
    )
    with torch.no_grad():
        maps = backbone(x)[0]
        top, left, bottom, right = s.shapes[CIRCLE].box
        h, w = bottom - top + 1, right - left + 1
        aligned = torch.zeros(64, KM, KM)
        points = []
        for i in range(KM):
            for j in range(KM):
                row = (top - 0.5 + (i + 0.5) * h / KM - 3.5) / 8
                col = (left - 0.5 + (j + 0.5) * w / KM - 3.5) / 8
                points.append((row, col))
                r, c = int(row), int(col)
                t, u = row - r, col - c
                above = (1 - u) * maps[:, r, c] + u * maps[:, r, c + 1]
                below = (1 - u) * maps[:, r + 1, c] + u * maps[:, r + 1, c + 1]
                aligned[:, i, j] = (1 - t) * above + t * below
        rows = [int(top + (i + 0.5) * h / M) for i in range(M)]
        cols = [int(left + (j + 0.5) * w / M) for j in range(M)]
        goal = torch.tensor(s.shapes[CIRCLE].mask[rows][:, cols]).float()
        logits = mask_head(aligned.unsqueeze(0))[0]
        p = 1 / (1 + torch.exp(-logits[0]))
        loss = -(goal * torch.log(p) + (1 - goal) * torch.log(1 - p)).mean()
        same = float((aligned - roi_align(maps, s.shapes[CIRCLE].box, KM)).abs().max())
    net = MaskRCNN()
    return {"maps": tuple(maps.shape), "box": tuple(s.shapes[CIRCLE].box), "size": (h, w),
            "first": tuple(round(v, 4) for v in points[0]), "last": tuple(round(v, 4) for v in points[-1]),
            "pool_squares": (int(top // 8), int(left // 8), int(bottom // 8), int(right // 8)),
            "aligned": tuple(aligned.shape), "row": tuple(round(v, 4) for v in aligned[0, 0].tolist()),
            "same_as_roi_align": same < 1e-5,
            "target": _FCN.pack(goal.numpy()), "ones": int(goal.sum()),
            "same_as_target": bool(torch.equal(goal, target(s.shapes[CIRCLE].mask, s.shapes[CIRCLE].box))),
            "logits": tuple(logits.shape), "p": (round(float(p.min()), 4), round(float(p.max()), 4)),
            "loss": round(float(loss), 4),
            "same_as_loss": abs(float(mask_loss(logits.unsqueeze(0), torch.tensor([0]), goal.unsqueeze(0))) - float(loss)) < 1e-5,
            "params": sum(v.numel() for v in net.parameters()),
            "mask_params": sum(v.numel() for v in net.mask_head.parameters())}


def measure() -> dict:
    """Every number REFERENCE holds, measured again: the worked example, and what the lesson says of the masks of the
    test scenes."""
    ss, _ = scene_set("test")
    return worked() | {"touching": touching(ss), "box_share": box_share(ss),
                       "round_trip": {m: round_trip(ss, m) for m in (4, 8, 14, 16, 28)}}


# --- pictures ---------------------------------------------------------------------------------------------------------

CROP = _ONE.CROP
ROI_CROP = (48, 40, 95, 87)     # the part of the scene of the worked example around the circle: 6 by 6 squares
low_high = _ONE.low_high


def unpacked(text: str) -> np.ndarray:
    """A mask that the lesson keeps as short text, as a bool array."""
    return _FCN.unpack(text, SIZE).astype(bool)


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _runs(a) -> list[tuple[int, int, int]]:
    """For every row of a bool array, its runs of True: (row, first, after the last)."""
    out = []
    for i, line in enumerate(np.asarray(a, dtype=bool)):
        c, n = 0, len(line)
        while c < n:
            if line[c]:
                end = c + 1
                while end < n and line[end]:
                    end += 1
                out.append((i, c, end))
                c = end
            else:
                c += 1
    return out


def _outline_path(mask) -> str:
    """The edge of a mask as one SVG path, in pixels: a line along every side of a pixel of the mask whose neighbor
    is not in it."""
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    parts = [f"M{c0} {r}H{c1}" for r, c0, c1 in _runs(m[1:, 1:-1] != m[:-1, 1:-1])]
    parts += [f"M{c} {r0}V{r1}" for c, r0, r1 in _runs((m[1:-1, 1:] != m[1:-1, :-1]).T)]
    return "".join(parts)


def instances_svg(s, title: str, width: int = 150) -> str:
    """The masks of the shapes of a scene, one for every object: each in the group color of its kind, with a line
    in ink around it, so that two objects that touch stay two."""
    n = SIZE
    scale = width / n
    top = _FCN._TOP
    parts = [_FCN._title(title, width), f'<g transform="translate(0,{top}) scale({scale:.5g})">',
             f'<rect x="0" y="0" width="{n}" height="{n}" style="fill:var(--ea-page)"/>',
             '<g shape-rendering="crispEdges">']
    for sh in s.shapes:
        parts += [f'<rect x="{c0}" y="{r}" width="{c1 - c0}" height="1" style="fill:var(--ea-group-{sh.label + 1})"/>'
                  for r, c0, c1 in _runs(sh.mask)]
    parts.append("</g>")
    for sh in s.shapes:
        parts.append(f'<path d="{_outline_path(sh.mask)}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.6;'
                     'vector-effect:non-scaling-stroke;stroke-linecap:square"/>')
    parts.append("</g>")
    parts.append(f'<rect x="0.5" y="{top + 0.5}" width="{width - 1}" height="{width - 1}" '
                 'style="fill:none;stroke:var(--ea-axis);stroke-width:1"/>')
    return _FCN._svg(parts, width, width + top)


def instance_picture(words: dict) -> str:
    """A test scene in which two circles touch: the scene, its class map, in which they are one area, and the masks of
    its objects, in which they are two."""
    s = scenes.scene(INSTANCE_SEED)
    panels = [(_FCN.photo_svg(INSTANCE_SEED, words["scene"]), 150), (_FCN.map_svg(_FCN.class_map(s), words["classes"]), 150),
              (instances_svg(s, words["objects"]), 150)]
    return _FCN.side_by_side(panels, words["alt"]) + _FCN.key_html(words["kinds"])


def _faded(img) -> np.ndarray:
    """A scene in light grey, so that masks in the color of a prediction stand out on every shape."""
    grey = np.asarray(img, dtype=float).mean(axis=2, keepdims=True)
    return np.repeat(255 - (255 - grey) * 0.55, 3, axis=2).astype(np.uint8)


def _masks_svg(img, found: list, truth: list, title: str, figsize=(3.9, 4.05), crop=CROP, size=10, fits=None) -> str:
    """A scene with masks over it: each mask found filled in the color of a prediction, with a line around it, or with
    `fits`, only the masks that fit filled and the others as a dashed line; the edge of each true mask in the color
    of data, with a thin edge in the color of the page."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    fig, ax = plt.subplots(figsize=figsize)
    _ONE._photo_axes(ax, _faded(img), title, crop=crop, size=size)
    for q, mask in enumerate(found):
        m = np.asarray(mask, dtype=float)
        if not m.any():
            continue
        if fits is None or fits[q]:
            ax.contourf(m, levels=[0.5, 1.5], colors=[plots.PRED], alpha=0.55)
            ax.contour(m, levels=[0.5], colors=[plots.PRED], linewidths=1.3)
        else:
            ax.contour(m, levels=[0.5], colors=[plots.PAGE], linewidths=3.0)
            ax.contour(m, levels=[0.5], colors=[plots.PRED], linewidths=1.5, linestyles=[(0, (3, 1.6))])
    for mask in truth:
        m = np.asarray(mask, dtype=float)
        if m.any():
            ax.contour(m, levels=[0.5], colors=[plots.PAGE], linewidths=2.6)
            ax.contour(m, levels=[0.5], colors=[plots.DATA], linewidths=1.1)
    svg = plots.render(fig)
    plt.close(fig)
    return svg


def stages_picture(words: dict) -> str:
    """The scene of the worked example and the course's trained Mask R-CNN (seed 0): its boxes of 0.5 or more, after
    NMS, with their kind and score; and the mask the mask head draws in each, pasted into the scene, with the edges of
    the true masks."""
    s = example()
    found = TRAINED["scene"]
    left = _ONE._scene_svg(s, [(k, sc, b) for k, sc, b, _ in found], words["boxes"].format(n=len(found)), names=words)
    right = _masks_svg(s.image, [unpacked(t) for *_, t in found], [sh.mask for sh in s.shapes], words["masks"].format(n=len(found)))
    return _ONE._side_by_side([(left, 300), (right, 300)], words["alt"])


def roi_picture(words: dict) -> str:
    """The circle's box on the scene of the worked example, cut to 8 by 8 bins for the mask head: by RoI pooling, the
    squares its corners lie in (solid), with a line where a bin of squares starts; by RoIAlign, its bins (thin lines)
    and the point in the middle of each. Both with the squares of 8 by 8 pixels as thin lines, and the box dashed."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    s = example()
    box = s.shapes[CIRCLE].box
    top, left, bottom, right = box
    r0, c0, r1, c1 = int(top // 8), int(left // 8), int(bottom // 8), int(right // 8)
    panels = []
    for which in ("pool", "align"):
        plots.style()
        fig, ax = plt.subplots(figsize=(2.6, 2.75))
        _ONE._photo_axes(ax, s.image, words[which], crop=ROI_CROP, size=9)
        for k in range(1, SIZE // STRIDE):   # the squares of the feature maps, 8 by 8 pixels
            ax.axhline(k * STRIDE - 0.5, color=plots.PAGE, linewidth=0.7, alpha=0.8)
            ax.axvline(k * STRIDE - 0.5, color=plots.PAGE, linewidth=0.7, alpha=0.8)
        if which == "pool":   # 8 bins of 4 squares: every square is two bins, and each bin a whole square
            _ONE._draw_box(ax, (r0 * 8, c0 * 8, (r1 + 1) * 8 - 1, (c1 + 1) * 8 - 1), plots.INK, "-", width=2.0, halo=4.0)
            for colour, width in ((plots.PAGE, 2.4), (plots.INK, 0.9)):
                for q in range(r0 + 1, r1 + 1):
                    ax.plot([c0 * 8 - 0.5, (c1 + 1) * 8 - 0.5], [q * 8 - 0.5] * 2, color=colour, linewidth=width)
                for q in range(c0 + 1, c1 + 1):
                    ax.plot([q * 8 - 0.5] * 2, [r0 * 8 - 0.5, (r1 + 1) * 8 - 0.5], color=colour, linewidth=width)
        else:
            h, w = bottom - top + 1, right - left + 1
            for colour, width in ((plots.PAGE, 2.0), (plots.INK, 0.6)):   # where the bins meet
                for i in range(1, KM):
                    ax.plot([left - 0.5, right + 0.5], [top - 0.5 + i * h / KM] * 2, color=colour, linewidth=width)
                    ax.plot([left - 0.5 + i * w / KM] * 2, [top - 0.5, bottom + 0.5], color=colour, linewidth=width)
            ys, xs = bin_middles(top, h, KM), bin_middles(left, w, KM)
            ax.plot([x for _ in ys for x in xs], [y for y in ys for _ in xs], linestyle="none", marker="o", markersize=3.2,
                    color=plots.INK, markeredgecolor=plots.PAGE, markeredgewidth=0.8)
        _ONE._draw_box(ax, box, plots.DATA, (0, (3, 1.6)))
        panels.append((plots.render(fig), 200))
        plt.close(fig)
    return _ONE._side_by_side(panels, words["alt"])


def scene_ious(found: list) -> list[float]:
    """For the masks of the scene of the worked example, the highest score first: the IoU of each with the shape of
    its kind that it fits best, of those that no mask with a higher score took first; 0 when none is left."""
    s = example()
    taken, out = set(), []
    for k, _, _, text in found:
        mk = unpacked(text)
        best, shape = 0.0, None
        for q, sh in enumerate(s.shapes):
            if sh.label == k and q not in taken and mask_iou(mk, sh.mask) > best:
                best, shape = mask_iou(mk, sh.mask), q
        if shape is not None:
            taken.add(shape)
        out.append(round(best, 4))
    return out


def threshold_frame(t: float, words: dict) -> tuple[str, dict]:
    """For the slider: the masks of the course's Mask R-CNN (seed 0) on the scene of the worked example, trained with
    RoI pooling and with RoIAlign: filled when their IoU with their shape is above t, else dashed; next to them the
    mask AP of both on the 200 test scenes at every threshold."""
    s = example()
    point = words["point"]
    panels, stop = [], {}
    for name, key in (("pool", "pool_scene"), ("align", "scene")):
        found = TRAINED[key]
        fits = [v > t for v in scene_ious(found)]
        stop[name] = {"ap": TRAINED["thresholds"][name][t], "fit": sum(fits), "masks": len(found)}
        panels.append((_masks_svg(s.image, [unpacked(x[3]) for x in found], [sh.mask for sh in s.shapes],
                                  words[name].format(n=sum(fits), t=f"{t:.2f}".replace(".", point)), figsize=(2.6, 2.75), size=9,
                                  fits=fits), 190))
    xs = list(THRESHOLDS)
    p = chart.Panel((0.48, 0.97), (0, 1.0), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], title=words["chart"], xlabel=words["x"],
                    ylabel=words["y"], xticks=[0.5, 0.6, 0.7, 0.8, 0.9], xtick_labels=[f"{v:.1f}".replace(".", point) for v in (0.5, 0.6, 0.7, 0.8, 0.9)])
    p.line(xs, [TRAINED["thresholds"]["align"][v] for v in xs], "ink", 2.2)
    p.line(xs, [TRAINED["thresholds"]["pool"][v] for v in xs], "ink", 2.2, dashed=True)
    p.points([t, t], [stop["align"]["ap"], stop["pool"]["ap"]], "ink", r=4.5)
    p.text(0.53, 0.06, words["pool_line"], "ink", "start")   # under both lines, at the left
    p.text(0.53, 0.16, words["align_line"], "ink", "start")
    plot = chart.figure([p], alt=words["chart_alt"], width=276, height=230)   # 11 px text in a play box on a phone
    if point == ",":
        plot = chart.with_decimal_comma(plot)
    return _ONE._side_by_side(panels + [(plot, 276)], words["alt"].format(t=f"{t:.2f}".replace(".", point))), stop


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss of each part after every epoch of the demo's run: the RPN's solid, the second stage's dashed, the mask
    head's solid with rings, with a key that says so. With `at`, a line at that epoch, for rewinding."""
    rpn, second, mask = curves.get("rpn", []), curves.get("second", []), curves.get("mask", [])
    mine = [v for v in rpn + second + mask if np.isfinite(v)]
    top = max([0.2] + [1.1 * v for v in mine])
    n = max(len(rpn), len(second), len(mask), 1)
    p = chart.Panel((0.5, n + 0.5), (0, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    xs = list(range(1, n + 1))
    if second:
        p.line(xs[:len(second)], second, "loss", 2.2, dashed=True)
    if rpn:
        p.line(xs[:len(rpn)], rpn, "loss", 2.2)
    if mask:
        p.line(xs[:len(mask)], mask, "loss", 2.2)
        p.points(xs[:len(mask)], mask, "loss", r=3.2, hollow=True)
    for q, name in enumerate((labels["rpn"], labels["second"], labels["mask"])):   # the key, top right, above the lines
        p.text(n + 0.4, top * (0.92 - 0.12 * q), name, "loss", "end")
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=276, height=220)   # 11 px text in a play box on a phone
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_picture(moments: list, words: dict) -> str:
    """The test scenes of the rewind, each with the edges of its true masks and the masks of the boxes of 0.5 or more
    that the learner's Mask R-CNN gave it at one moment. words["scene"] is one label, or two: for one, and for any
    other number."""
    panels = []
    for seed, found in zip(SHOWN, moments):
        s = scenes.scene(seed)
        panels.append((_masks_svg(s.image, [unpacked(t) for *_, t in found], [sh.mask for sh in s.shapes],
                                  (words["scene"] if isinstance(words["scene"], str) else words["scene"][0 if len(found) == 1 else 1]).format(seed=seed, n=len(found)),
                                  figsize=(2.3, 2.45), crop=(0, 0, 127, 127), size=9), 160))
    return _ONE._side_by_side(panels, words["alt"])


# --- the demo: the course's Mask R-CNN with the learner's roi_align and paste ----------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


_Stopped, _named = _TWO._Stopped, _TWO._named


def _demo(show=None):
    """For the demo: None when roi_align.py or paste.py is not in the project yet; "roi_align" when the learner's
    roi_align (or the bilinear it uses) stops with an error or does not give (64, k, k) numbers that training can follow
    back for a box of a scene; "paste" when their paste stops or does not give a mask of the scene; "proposals" when
    their proposals stops or does not give boxes; "offsets" when their boxes_from stops on the trained detector's
    offsets; "slow" when the training takes too long; else a dict with the loss of each part after every epoch
    ("curves"), the boxes and masks on the SHOWN scenes after every epoch ("moments"), the layer the demo watches after
    every epoch ("layers", see look), the summary of the trained Mask R-CNN on the test scenes, its parameters, and the
    modules the course's versions stood in for. `show` hears the run while it trains, every report with "look": a
    function that gives what Mask R-CNN makes at that moment (decision 1.17)."""
    if not (_saved("roi_align") and _saved("paste")):
        return None
    names = {}
    for module, name in (("roi_align", "roi_align"), ("paste", "paste"), ("proposals", "proposals"), ("offsets", "boxes_from")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return module
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    ss, X = scene_set("train")
    torch.manual_seed(0)
    model = MaskRCNN()
    try:   # one scene through all: the shapes the training needs
        maps, rows = model(X[:1].float() / 255)
        props = as_boxes(names["proposals"](rows[0].detach(), anchors().clone(), TRAIN_KEEP))
        if not (1 <= len(props) <= TRAIN_KEEP and torch.isfinite(props).all()):
            return "proposals"
    except Exception:
        return "proposals"
    box = tuple(float(v) for v in boxes_and_labels(ss[0])[0][0].tolist())
    try:
        for k in (K, KM):
            p = names["roi_align"](maps[0], box, k)
            if not (torch.is_tensor(p) and tuple(p.shape) == (64, k, k) and p.requires_grad and torch.isfinite(p).all()):
                return "roi_align"
    except Exception:
        return "roi_align"
    try:
        m = torch.as_tensor(names["paste"](torch.rand(M, M, generator=torch.Generator().manual_seed(0)), box, SIZE))
        if tuple(m.shape) != (SIZE, SIZE):
            return "paste"
    except Exception:
        return "paste"
    align, props_of, boxes_of, paste_of = (_named(names["roi_align"], "roi_align"), _named(names["proposals"], "proposals"),
                                           _named(names["boxes_from"], "offsets"), _named(names["paste"], "paste"))
    torch.manual_seed(0)   # the same start as the course's run
    model = MaskRCNN()
    test_ss, Xt = scene_set("test")
    example_x = Xt[TEST.index(SEED):TEST.index(SEED) + 1].float() / 255   # the scene of the worked example
    moments, layers, broken = [], [], []

    def look() -> dict:
        """What Mask R-CNN makes at this moment: its masks on the SHOWN test scenes, and the last layer of its
        backbone, the feature maps all three parts read, on the scene of the worked example; looked at in eval mode
        without a graph, so the run stays the same."""
        with netview.looking(model):
            masks = shown_moment(model, align, props_of, boxes_of, paste_of)
        return {"masks": masks, "layer": netview.watched(model, example_x)}

    def heard(p):
        if "model" in p:
            try:
                moments.append(shown_moment(p["model"], align, props_of, boxes_of, paste_of))
            except _Stopped as e:
                broken.append(e.key)
                moments.append([[] for _ in SHOWN])
            except Exception:
                broken.append("offsets")
                moments.append([[] for _ in SHOWN])
            layers.append(netview.watched(model, example_x))
            p = {k: v for k, v in p.items() if k != "model"}
        if show is not None:
            try:
                show(p | {"look": look})
            except Timeout:   # the demo's time limit, also when it falls while the chart draws
                raise
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass

    try:   # with_time_limit trains on one thread, as the course's run
        history = with_time_limit(lambda: train(model, 0, EPOCHS, heard, align, props_of), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    except Exception:   # the numbers of a box did not fit a head
        return "roi_align"
    if broken:
        return broken[0]
    try:
        result = summary(detect(model, Xt, TEST_KEEP, align, props_of, boxes_of, paste_of), test_ss)
    except _Stopped as e:
        return e.key
    except Exception:
        return "paste"
    curves = {key: [float(h[key]) for h in history] for key in ("rpn", "second", "mask")}
    return result | {"curves": curves, "moments": moments, "layers": layers, "params": sum(v.numel() for v in model.parameters()),
                     "stand_ins": stand_ins}


def now_pictures(seen: dict, words: dict, lang: str, uid: str) -> list[str]:
    """What Mask R-CNN makes at one moment of the demo's run, as `look` or a rewind gives it: its masks on the SHOWN
    test scenes, and the layer the demo watches, under words["layer"] with its name."""
    return netview.moment(rewind_picture(seen["masks"], words), seen.get("layer"), words["layer"], lang, uid, seed=SEED)


def epoch_frame(result: dict, e: int, labels: dict, words: dict, lang: str) -> str:
    """The rewind at epoch e (from 1): the loss of each part after every epoch with a line at e, then the masks on
    the SHOWN test scenes after it, and under them the layer the demo watches, as it was then."""
    seen = {"masks": result["moments"][e - 1], "layer": result["layers"][e - 1]}
    return live.rewound(curves_picture(result["curves"], labels, lang, at=e), now_pictures(seen, words, lang, f"nv-mask-epoch-{e}"))


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("roi_align", "paste", "bilinear", "proposals", "offsets"):
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

def _measured(fn, seeds) -> tuple[dict, dict]:
    """The course's Mask R-CNN trained with every seed, with `fn` cutting the boxes: the summary and the curves of
    each, and the pictures of the first."""
    runs, pictures = {}, None
    ss, Xt = scene_set("test")
    for seed in seeds:
        model, history = train_course(seed, fn)
        runs[seed] = summary(detect(model, Xt, TEST_KEEP, fn), ss)
        runs[seed]["curves"] = {key: [round(h[key], 4) for h in history] for key in ("rpn", "second", "mask")}
        if seed == min(seeds):
            pictures = pictures_of(model, fn)
    return runs, pictures


def measure_trained(seeds=range(5)) -> dict:
    """Everything TRAINED holds, measured again: every seed with RoIAlign and with RoI pooling, and with the first seed
    the pictures of both. On one thread, a seed takes about eleven minutes with RoIAlign and five with RoI pooling."""
    keys = ("map", "aps", "map75", "box_map", "box_map75", "found", "false", "median")
    align, align_pictures = _measured(roi_align, seeds)
    pool, pool_pictures = _measured(roi_pool, seeds)
    first = min(seeds)
    out = {key: [align[seed][key] for seed in seeds] for key in keys}
    return out | {"shapes": align[first]["shapes"], "curves": align[first]["curves"], "scene": align_pictures["scene"],
                  "pool": {key: [pool[seed][key] for seed in seeds] for key in keys}, "pool_scene": pool_pictures["scene"],
                  "thresholds": {"align": align_pictures["thresholds"], "pool": pool_pictures["thresholds"]}}
