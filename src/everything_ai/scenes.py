# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
"""The course's own scenes (course/route/bronnen.yaml, course-scenes): small pictures of circles, squares and
triangles on a grey background, with the right annotation of every shape: its kind, its bounding box, its mask and,
for squares and triangles, its corners as keypoints. The course makes them itself, so they show no people, raise no
licence questions, and the course may train models on them and share those. The same seed always gives the same
scene, on every computer.

A shape drawn later lies on top of the earlier ones and can hide part of them. A mask holds the pixels of the shape
that can be seen, and its box is the smallest box around that mask. A keypoint is (row, column, v): the pixel of the
whole shape nearest to a corner, with v as COCO writes it: 2 when that pixel is in the mask, 1 when it is hidden under
a later shape. The corners go round clockwise, as the image shows them, from the corner that the shape's turn puts
first. Boxes follow the course: (top, left, bottom, right), from the smallest to the largest row and column, both
included."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SIZE = 128                                  # a scene is SIZE by SIZE pixels
KINDS = ("circle", "square", "triangle")    # the label of a shape is its place in KINDS
CORNERS = {"circle": 0, "square": 4, "triangle": 3}   # its keypoints: a circle has none
SHAPES = (3, 5)                             # a scene has from 3 to 5 shapes
RADIUS = (10.0, 22.0)                       # from the centre of a shape to its edge (circle) or its corners
SMALLEST = 60                               # every shape keeps at least this many pixels in sight
TRIES = 1000                                # scenes drawn for one seed before `scene` gives up
# Colours that stand out from the grey background and from each other; a shape gets one at random, so that its
# colour does not give its kind away.
COLOURS = ((214, 69, 65), (52, 120, 196), (60, 160, 90), (232, 176, 40), (140, 86, 180), (40, 170, 175))
_BACKGROUND = 222      # the grey of the background, with a little noise and a soft slope across it


@dataclass
class Shape:
    kind: str
    label: int
    box: tuple[int, int, int, int]     # (top, left, bottom, right) of its mask, both ends included
    mask: np.ndarray                   # bool, SIZE by SIZE: the pixels of the shape that can be seen
    keypoints: list[tuple[int, int, int]] = field(default_factory=list)   # (row, column, v) per corner, clockwise
    centre: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    angle: float = 0.0                 # its turn in radians, as `shape_mask` takes it
    area: int = 0                      # the number of pixels of the whole shape, hidden ones included

    @property
    def whole(self) -> bool:
        """Whether no later shape hides any of it."""
        return int(self.mask.sum()) == self.area


@dataclass
class Scene:
    seed: int
    image: np.ndarray                  # uint8, SIZE by SIZE by 3
    shapes: list[Shape]


def _corners(kind: str, centre: tuple[float, float], radius: float, angle: float) -> list[tuple[float, float]]:
    """The corners of a square or a triangle, as (row, column), turned by angle; none for a circle."""
    k = CORNERS[kind]
    return [(centre[0] - radius * math.cos(angle + 2 * math.pi * i / k),
             centre[1] + radius * math.sin(angle + 2 * math.pi * i / k)) for i in range(k)]


def _inside(kind: str, centre, radius: float, corners, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The pixels whose centre lies inside the shape."""
    if kind == "circle":
        return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2
    inside = np.ones(rows.shape, dtype=bool)
    for (r0, c0), (r1, c1) in zip(corners, corners[1:] + corners[:1]):
        # on the same side of every edge as the centre
        side = (c1 - c0) * (rows - r0) - (r1 - r0) * (cols - c0)
        centre_side = (c1 - c0) * (centre[0] - r0) - (r1 - r0) * (centre[1] - c0)
        inside &= side * centre_side >= 0
    return inside


def shape_mask(kind: str, centre: tuple[float, float], radius: float, angle: float, size: int = SIZE):
    """The pixels of one shape alone, turned by `angle` (radians), and its corners: as `scene` draws it."""
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    corners = _corners(kind, centre, radius, angle)
    return _inside(kind, centre, radius, corners, rows, cols), corners


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    rows, cols = np.mgrid[0:size, 0:size]
    slope = rng.uniform(-6, 6, 2)
    grey = _BACKGROUND + slope[0] * (rows / size - 0.5) + slope[1] * (cols / size - 0.5) + rng.normal(0, 3, (size, size))
    return np.repeat(np.clip(grey, 0, 255)[:, :, None], 3, axis=2)


def scene(seed: int, size: int = SIZE, count: int | None = None) -> Scene:
    """The scene of this seed: from 3 to 5 shapes, or `count`, each at least SMALLEST pixels in sight."""
    if size < 2 * RADIUS[1] + 4:
        raise ValueError(f"a scene needs at least {int(2 * RADIUS[1] + 4)} by {int(2 * RADIUS[1] + 4)} pixels for its largest shape")
    if count is not None and count < 1:
        raise ValueError("a scene has at least one shape")
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    for _ in range(TRIES):
        image = _background(rng, size)
        owner = np.full((size, size), -1)
        drawn = []
        n = int(rng.integers(SHAPES[0], SHAPES[1] + 1))
        for i in range(count if count is not None else n):
            kind = KINDS[int(rng.integers(len(KINDS)))]
            radius = float(rng.uniform(*RADIUS))
            centre = (float(rng.uniform(radius + 1, size - radius - 2)), float(rng.uniform(radius + 1, size - radius - 2)))
            angle = float(rng.uniform(0, 2 * math.pi))
            corners = _corners(kind, centre, radius, angle)
            colour = COLOURS[int(rng.integers(len(COLOURS)))]
            pixels = _inside(kind, centre, radius, corners, rows, cols)
            owner[pixels] = i
            noise = rng.normal(0, 4, (int(pixels.sum()), 3))
            image[pixels] = np.clip(np.array(colour) + noise, 0, 255)
            drawn.append((kind, centre, radius, angle, corners, pixels))
        if all((owner == i).sum() >= SMALLEST for i in range(len(drawn))):
            break
    else:
        raise ValueError(f"no scene of {count} shapes keeps {SMALLEST} pixels of each in sight; try fewer shapes")
    shapes = []
    for i, (kind, centre, radius, angle, corners, pixels) in enumerate(drawn):
        mask = owner == i
        r, c = np.nonzero(mask)
        pr, pc = np.nonzero(pixels)
        keypoints = []
        for cr, cc in corners:
            # the pixel of the whole shape nearest to the corner; seen when no later shape lies on it
            j = int(np.argmin((pr - cr) ** 2 + (pc - cc) ** 2))
            kr, kc = int(pr[j]), int(pc[j])
            keypoints.append((kr, kc, 2 if mask[kr, kc] else 1))
        shapes.append(Shape(kind, KINDS.index(kind), (int(r.min()), int(c.min()), int(r.max()), int(c.max())), mask,
                            keypoints, centre, radius, angle, int(pixels.sum())))
    return Scene(seed, image.astype(np.uint8), shapes)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """The intersection over union of two boxes, as cv-iou computes it."""
    def area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    both = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1) * max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    return both / (area(a) + area(b) - both)


def proposals(s: Scene, seed: int, per: int = 7) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Boxes that a detector could propose for the shapes of a scene, before non-maximum suppression, and a score for
    each box. Every shape gets `per` boxes: its own box, moved and stretched a little at random, each with an IoU of at
    least 0.55 with it. A score lies between 0 and 1, with two decimals, and no two are the same: it is mostly higher the
    better the box fits, with a little noise, and the shapes differ in how high it gets, as a detector is surer of one
    object than of another.
    The boxes come in a random order. The course makes them itself, so that a lesson can show what happens after a
    detector before it has one: they are not a real detector's output. The same scene and seed always give the same
    boxes."""
    if per * len(s.shapes) > 95:
        raise ValueError("there are only 95 different scores with two decimals from 0.05 to 0.99")
    rng = np.random.default_rng(seed)
    size = s.image.shape[0]
    made = []
    for k, shape in enumerate(s.shapes):
        top, left, bottom, right = shape.box
        h, w = bottom - top + 1, right - left + 1
        peak = rng.uniform(0.75, 0.97)            # the highest score this shape can get
        count, tries = 0, 0
        while count < per:
            tries += 1
            if tries > TRIES * per:
                raise ValueError(f"no {per} different boxes with an IoU of at least 0.55 for shape {k}; try fewer")
            cy = (top + bottom) / 2 + rng.normal(0, 0.07 * h)
            cx = (left + right) / 2 + rng.normal(0, 0.07 * w)
            nh = h * math.exp(rng.normal(0, 0.07))
            nw = w * math.exp(rng.normal(0, 0.07))
            box = (max(0, int(round(cy - (nh - 1) / 2))), max(0, int(round(cx - (nw - 1) / 2))),
                   min(size - 1, int(round(cy + (nh - 1) / 2))), min(size - 1, int(round(cx + (nw - 1) / 2))))
            fit = _iou(box, shape.box)
            if fit < 0.55:
                continue
            score = round(float(np.clip(peak * fit ** 1.5 + rng.normal(0, 0.03), 0.05, 0.99)), 2)
            if any(score == m[1] for m in made):
                continue
            made.append((box, score))
            count += 1
    order = rng.permutation(len(made))
    return [made[i][0] for i in order], [made[i][1] for i in order]


MISS, WRONG, SECOND, EXTRA = 0.1, 0.08, 0.12, 0.4   # how often `detections` misses a shape, names the wrong kind,
SPREAD = 0.08                                       # adds a looser second box, puts a box on the background; how far
                                                    # it moves and stretches a box


def detections(scenes: list[Scene], seed: int) -> list[tuple[int, str, float, tuple[int, int, int, int]]]:
    """What a detector of middling quality could give for a list of scenes, after non-maximum suppression with a
    threshold of 0.5 for each kind: for each box, (image, kind, score, box), with image the place of its scene in the
    list. Most shapes get one box of their own kind: their own box, moved and stretched a little at random. Now and
    then a shape gets no box, a box of another kind, or a second, looser box as well, and a box lies on the
    background. A score lies between 0 and 1, with two decimals, and no two are the same: it is mostly higher the better
    the box fits, with noise. The list comes sorted by score, highest first. There are only 95 such scores, so the
    boxes of all scenes together can be at most 95 (about 20 scenes); with more, it stops with a ValueError.
    The course makes these boxes itself, so that a lesson can score a detector before it has one: they are not a real
    detector's output. The same scenes and seed always give the same boxes."""
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    made = []

    def score(x: float) -> float:
        # the nearest score with two decimals from 0.05 to 0.99 that no box has yet
        free = [v for v in range(5, 100) if v not in used]
        if not free:
            raise ValueError("there are only 95 different scores with two decimals from 0.05 to 0.99")
        v = min(free, key=lambda f: (abs(f / 100 - x), f))
        used.add(v)
        return v / 100

    for image, s in enumerate(scenes):
        size = s.image.shape[0]

        def moved(box, spread):
            top, left, bottom, right = box
            h, w = bottom - top + 1, right - left + 1
            cy = (top + bottom) / 2 + rng.normal(0, spread * h)
            cx = (left + right) / 2 + rng.normal(0, spread * w)
            nh, nw = h * math.exp(rng.normal(0, spread)), w * math.exp(rng.normal(0, spread))
            # every side stays inside the image, so that top <= bottom and left <= right
            top = min(size - 1, max(0, int(round(cy - (nh - 1) / 2))))
            left = min(size - 1, max(0, int(round(cx - (nw - 1) / 2))))
            return (top, left, max(top, min(size - 1, int(round(cy + (nh - 1) / 2)))),
                    max(left, min(size - 1, int(round(cx + (nw - 1) / 2)))))

        boxes = []
        for shape in s.shapes:
            if rng.random() < MISS:
                continue
            box = moved(shape.box, SPREAD)
            kind = shape.kind
            if rng.random() < WRONG:
                kind = KINDS[(shape.label + int(rng.integers(1, len(KINDS)))) % len(KINDS)]
            base = 0.25 + 0.7 * _iou(box, shape.box) ** 2 + rng.normal(0, 0.1)
            boxes.append((kind, base, box))
            if rng.random() < SECOND:
                boxes.append((shape.kind, base - rng.uniform(0.1, 0.4), moved(shape.box, 2.5 * SPREAD)))
        for _ in range(int(rng.poisson(EXTRA))):
            for _ in range(TRIES):
                h, w = int(rng.integers(14, 40)), int(rng.integers(14, 40))
                top, left = int(rng.integers(0, size - h + 1)), int(rng.integers(0, size - w + 1))
                box = (top, left, top + h - 1, left + w - 1)
                if all(_iou(box, shape.box) <= 0.1 for shape in s.shapes):
                    break
            else:
                raise ValueError(f"no room on the background of scene {s.seed} for a box")
            boxes.append((KINDS[int(rng.integers(len(KINDS)))], float(rng.uniform(0.05, 0.6)), box))
        # non-maximum suppression with a threshold of 0.5, for each kind on its own
        boxes.sort(key=lambda b: -b[1])
        kept = []
        for kind, base, box in boxes:
            if all(k != kind or _iou(box, b) <= 0.5 for k, _, b in kept):
                kept.append((kind, base, box))
        made += [(image, kind, base, box) for kind, base, box in kept]
    made.sort(key=lambda d: -d[2])     # the surest boxes first, so that they get the score nearest to their own
    out = [(image, kind, score(base), box) for image, kind, base, box in made]
    out.sort(key=lambda d: -d[2])
    return out


def anchor_boxes(cells: int = 16, stride: int = 8,
                 sizes: tuple[tuple[float, float], ...] = ((16, 16), (24, 24), (32, 32))) -> np.ndarray:
    """Anchor boxes for a scene, as the worked example of cv-anchors lays them: a feature map of `cells` by `cells`
    squares, each standing for `stride` by `stride` pixels, and around the middle of every square one anchor of each
    (height, width) in `sizes`. Row by row, square by square, size by size: an array of shape
    (cells * cells * len(sizes), 4), each row a box as the course writes it, (top, left, bottom, right) with both ends
    included, in floats. Anchors at the edge stick out of the scene."""
    rows = []
    for i in range(cells):
        for j in range(cells):
            y, x = i * stride + (stride - 1) / 2, j * stride + (stride - 1) / 2
            for h, w in sizes:
                rows.append((y - (h - 1) / 2, x - (w - 1) / 2, y + (h - 1) / 2, x + (w - 1) / 2))
    return np.array(rows, dtype=float).reshape(-1, 4)


def match_anchors(anchors, boxes) -> np.ndarray:
    """What each anchor learns, as `match` of cv-anchors tells it from the IoU of every anchor with every box (rows
    (top, left, bottom, right), both ends included): the number of the box it overlaps most from an IoU of 0.5 on,
    -2 (ignored) from 0.4 up to 0.5, -1 (background) below 0.4; and last, every box its best anchors, ties included,
    even below 0.5. An array of whole numbers, one per anchor."""
    anchors, boxes = np.asarray(anchors, dtype=float), np.asarray(boxes, dtype=float)
    ious = np.zeros((len(anchors), len(boxes)))
    for k, box in enumerate(boxes):
        top, left = np.maximum(anchors[:, 0], box[0]), np.maximum(anchors[:, 1], box[1])
        bottom, right = np.minimum(anchors[:, 2], box[2]), np.minimum(anchors[:, 3], box[3])
        overlap = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)
        area_a = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
        ious[:, k] = overlap / (area_a + (box[2] - box[0] + 1) * (box[3] - box[1] + 1) - overlap)
    best, most = ious.argmax(axis=1), ious.max(axis=1)
    label = np.full(len(anchors), -1)
    label[most >= 0.5] = best[most >= 0.5]
    label[(most >= 0.4) & (most < 0.5)] = -2
    for k in range(len(boxes)):
        label[ious[:, k] == ious[:, k].max()] = k
    return label


def nms(boxes, scores, threshold: float) -> list[int]:
    """Non-maximum suppression, as `nms` of cv-nms does it: the numbers of the boxes kept, the highest score first. A
    box goes when a box with a higher score that is kept overlaps it by more than `threshold` (the IoU of cv-iou, both
    ends included); of two equal scores, the first box counts as the higher. `boxes` has a row (top, left, bottom,
    right) for every box and `scores` a number for every box: lists, arrays or tensors. The IoU of every pair is worked
    out at once, so that a few hundred boxes take no time."""
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    s = np.asarray(scores, dtype=float).reshape(-1)
    top, left = np.maximum(b[:, None, 0], b[None, :, 0]), np.maximum(b[:, None, 1], b[None, :, 1])
    bottom, right = np.minimum(b[:, None, 2], b[None, :, 2]), np.minimum(b[:, None, 3], b[None, :, 3])
    both = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    ious = both / (area[:, None] + area[None, :] - both)
    gone, kept = np.zeros(len(b), dtype=bool), []
    for i in np.argsort(-s, kind="stable"):
        if not gone[i]:
            kept.append(int(i))
            gone |= ious[i] > threshold
    return kept
