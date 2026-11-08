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
