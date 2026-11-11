# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-fpn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-fpn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) and [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lateral():
    quick_lateral = L.choice(
        "quick-lateral",
        question="A 1×1 convolution from 128 channels to 32, `nn.Conv2d(128, 32, 1)`. How many parameters does it have?",
        options={
            "right": "4,128: 32 filters, each with 128 weights and a bias",
            "bias": "4,096: 128 × 32 weights",
            "three": "36,896: 32 filters, each with 9 × 128 weights and a bias",
            "sum": "160: 128 + 32",
        },
        correct="right",
        why={
            "right": "Each of the 32 filters has a weight for every channel that comes in, and a bias: 32 × (128 + 1) = 4,128. Today a 1×1 convolution like this gives every map of the backbone the same 32 channels.",
            "bias": "The weights are right, but every filter also has a bias: 32 × (128 + 1). See [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en).",
            "three": "That is a 3×3 convolution, with 9 weights per channel. A 1×1 convolution has one weight per channel that comes in: 32 × (128 + 1). See [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en).",
            "sum": "Every filter sees all 128 channels: 128 weights and a bias for each of the 32 filters, 32 × (128 + 1).",
        },
    )
    quick_lateral
    return (quick_lateral,)


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="The head of your one-stage detector gave a scene maps of shape (1, 24, 16, 16): 3 anchors times 8 numbers, on 16 by 16 squares. A head with 2 anchors per square, on maps of 32 by 32 squares, gives (1, 16, 32, 32). What shape do its rows get with `permute(0, 2, 3, 1)` and `reshape(1, -1, 8)`?",
        options={
            "right": "(1, 2048, 8): 32 × 32 squares, 2 anchors each",
            "squares": "(1, 1024, 8): one row for every square",
            "sixteen": "(1, 1024, 16): the 16 numbers of every square in one row",
        },
        correct="right",
        why={
            "right": "`permute` puts the 16 numbers of every square together, and `reshape(1, -1, 8)` cuts them into rows of 8: 32 × 32 × 2 = 2,048 anchors. Today every level gives its own rows, and they follow each other.",
            "squares": "A square holds 16 numbers, the 8 of each of its 2 anchors: rows of 8 make 2 rows per square, 32 × 32 × 2. See [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).",
            "sixteen": "`reshape(1, -1, 8)` makes rows of 8, one for every anchor: 32 × 32 × 2 = 2,048 of them. See [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your one-stage detector put all its anchors on one feature map, and its scenes had shapes of about one size; in a photo, a car can fill the picture, or be a few pixels in the distance. The old answer is an **image pyramid**: the same photo at several sizes, each smaller than the one before, with the detector on every one of them, which the paper of today's lesson says makes finding the objects of a photo about four times slower. A CNN already makes maps of several sizes on its way up, but the early ones, which are sharp, know little of what they see, and the deep ones, which know what is there, are coarse; SSD (2016), a one-stage detector, put anchors on several of them, but only from high up in the network on, and so missed the sharp maps that small objects need. A **feature pyramid network** (Lin and colleagues, 2017) gives the sharp maps what the deep ones know, with a way back down: today you build one, on scenes with small and large shapes.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_small():
    guess_small = L.guess(
        "small",
        question="The course trained a one-stage detector like yours, with one more part in its backbone, on scenes with small and large shapes: all its anchors on one map of 8 by 8 squares of 16 pixels, 8 sizes from 8 to 91 pixels on every square. The 200 test scenes have 268 small shapes, smaller than 16 by 16 pixels. How many of them did it find?",
        options={
            "all": "About 260: nearly all",
            "most": "About 200",
            "right": "About 130: half",
            "few": "About 30",
        },
        reveal="129 to 143, with the seeds 0 to 4: about half. Of the 236 large shapes, it found 218 to 225. A small shape is smaller than one square of that map, and the anchors of the squares lie 16 pixels apart: only 43 of the 268 small shapes have an anchor that overlaps them with an IoU of 0.5 or more. The feature pyramid network of this lesson finds 244 to 249 of them.",
    )
    guess_small
    return (guess_small,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Shapes of every size.** `scene(seed, spread=True)` gives scenes whose shapes have a radius from 5 to 40 pixels, spread evenly over the sizes: a shape twice as large as another is as likely as one half as large. Every shape keeps at least half its pixels in sight. The lesson calls a shape small when the square root of the area of its box, $\sqrt{wh}$, is below 16 pixels, and large from 32 pixels on; medium in between. The 200 test scenes have 786 shapes, from 7 to 80 pixels: 268 small, 282 medium and 236 large. COCO, the dataset of the paper, draws its lines at 32 and 96 pixels, in photos of about 640 pixels.

        **Four maps.** The backbone is the one of your one-stage detector with two more parts, each a max pooling and a convolution of 3 by 3 with batch normalization and ReLU. The paper calls the maps at the end of its four parts C2 to C5: C$_k$ has squares of $2^k$ pixels. C2 has 32 channels on 32 by 32 squares of 4 pixels; C3, the map your detector used, 64 on 16 by 16 squares of 8 pixels; C4 128 on 8 by 8 squares of 16 pixels; and C5 128 on 4 by 4 squares of 32 pixels. Every number of C2 sees 18 by 18 pixels of the scene, its receptive field; of C3 38, of C4 78, and of C5 158, more than the whole scene.

        **One map is not enough.** The course trained two detectors with all 8 anchor sizes on one map. On C4, the anchors of neighboring squares lie 16 pixels apart: a small shape between them overlaps none of them well, so only its best anchor learns it: 1.1 anchors per small shape on average, against 4.0 per large one. On C2, it is the other way around: a large shape is learned by 63 anchors on average, a small one by 4, and the 8,192 anchors give about 5 times as many false boxes as a pyramid. The table after the slider has their numbers.

        **A level for every size.** A feature pyramid gives every scale its own level, P2 to P5, from C2 to C5, and every level its own anchors: on each of its squares 2 anchors, 2 and $2\sqrt{2}$ squares wide. That is 8 and 11.3 pixels on P2, 16 and 22.6 on P3, 32 and 45.3 on P4, and 64 and 90.5 on P5: 2,048 + 512 + 128 + 32 = 2,720 anchors. Every anchor learns as in your one-stage detector, by its IoU with the true boxes, so a shape is learned by anchors of the level whose sizes fit it: 2.5 to 2.7 anchors per shape on average, small or large. The paper put its pyramid into a two-stage detector, Faster R-CNN, whose first stage, a *region proposal network*, proposes boxes that may hold an object, the *proposals*: [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en) explains it. Its anchors have one area per level, 32² to 512² pixels on P2 to P6, where P6 is P5 made half as large once more, each in 3 ratios of width to height, 1:2, 1:1 and 2:1: 15 kinds of anchor, 3 on every square. Here, the pyramid goes into your one-stage detector, as RetinaNet does in the next lesson.

        **The way down.** A **feature pyramid network** makes the levels from the top down, along what the paper calls the *top-down pathway*. P5 is C5 through a 1×1 convolution to 32 channels. Then, one level at a time: the level above made twice as large with nearest-neighbor interpolation, plus the map of the backbone through its own 1×1 convolution, the *lateral connection*. `F.interpolate(p, scale_factor=2, mode="nearest")` makes every number of `p` a block of 2 by 2. Last, a 3×3 convolution on every level, which the paper adds "to reduce the aliasing effect of upsampling": the sharp steps between the blocks of 2 by 2. So the sharp level P2 gets what C3, C4 and C5 know too. The paper uses 256 channels, and no ReLU in these layers; here 32 channels, and no ReLU either.

        **One head for every level.** One convolution of 3 by 3 gives every square of every level 2 × 8 numbers: 4 logits and 4 offsets for each of its 2 anchors. One head can serve all four levels, because they all have 32 channels and the anchors of every level are the same number of its squares wide: a shape 2 squares wide looks alike on every level. The rows of every level follow each other, the finest first, in the order of the anchors. In the paper, a head for every level gave about the same accuracy as one shared head; to the paper, that shows that after the way down, every level knows about as much of what it sees.

        **Hard negative mining.** In a training scene, on average 9.9 of the 2,720 anchors learn a shape, 0.4%; almost all the others learn background. Trained with the loss of your one-stage detector, the cross-entropy over all anchors, the FPN detector finds only 9 of the 786 shapes of the test scenes with a score of 0.5 or more: it learns to say background. **Hard negative mining**, as in SSD: in every scene, the cross-entropy counts all anchors that learn a shape, and three times as many background anchors, those with the highest cross-entropy, the ones the detector gets most wrong. The box loss is the mean absolute difference of the offsets, as before.
        """),
        L.watch_out("The rows and the anchors must be in the same order: level 2 first, then 3, 4 and 5; on every level row by row, square by square, and the 2 anchors of a square after each other. Rows in another order still give 2,720 rows of 8, and the detector still trains, but every row learns the targets of another anchor."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **feature pyramid network** (FPN) | maps of the same number of channels at every scale, made from the maps of the backbone from the top down: a 1×1 convolution of each, the level above made twice as large and added, then a 3×3 convolution | `pyramid` |
        | **image pyramid** | the same image at several sizes, each smaller than the one before, often half as large, to look for patterns at every scale | |
        | **lateral connection** | in an FPN, the 1×1 convolution that brings a map of the backbone to the level of the same size | `lateral[i]` |
        | **hard negative mining** | training on the negatives the model gets most wrong: in a detector, the background anchors with the highest loss, in SSD at most three for every anchor that learns an object | |
        | `F.interpolate(x, scale_factor=2, mode="nearest")` | makes every map of x twice as large each way, every number a block of 2 by 2 | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_pyramid():
    mo.vstack([
        mo.Html(C.pyramid_picture({
            "up": "backbone", "down": "feature pyramid",
            "head": ("one head", "for every level"),
            "alt": "On the left the maps C2 to C5 of the backbone, from bottom to top each half as large. From each map an arrow with 1×1 goes right. From the top down, each level goes twice as large into a plus with the 1×1 map of the level below. From each plus an arrow with 3×3 goes to P2 to P5, of 32 by 32 to 4 by 4 squares. A bracket on the right joins P2 to P5: one head for every level.",
        })),
        mo.md(r"""
        On the left, the backbone goes up: every map half as large as the one below it. In the middle, the way down: from the top, every level made twice as large, ×2, and added to the 1×1 map of the level below. On the right, after a 3×3 convolution, the levels P2 to P5, each with 32 channels, and one head for all of them.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Which level finds which shapes? The slider picks a level of the course's trained FPN detector, seed 0. On the left, the boxes of 0.5 or more on the scene of the worked example that come from anchors of that level; on the right, the small, medium and large shapes of the 200 test scenes that the detector finds: in light all of them, in dark those that come from that level.
    """)
    return


@app.cell(hide_code=True)
def explore_level_cell():
    def _frame(k):
        k = int(k)
        _html, _stop = C.level_frame(k, {"title": ("level {k}: {n} box", "level {k}: {n} boxes"), "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
                                         "chart": "found by level {k}", "shapes": "shapes", "thirds": ("small", "medium", "large"),
                                         "chart_alt": "Of the small, medium and large shapes the detector finds on the 200 test scenes, how many come from anchors of level {k}.",
                                         "alt": "The scene of the worked example with its five true boxes and the {n} dashed boxes of 0.5 or more that come from anchors of level {k}. Next to it, how many shapes of each size level {k} finds."})
        _found = _stop["found"]
        return _html, f"Level {k}: squares of {2 ** k} pixels, anchors of {L.number(_stop['sizes'][0], 1)} and {L.number(_stop['sizes'][1], 1)} pixels, {C.whole(_stop['anchors'], L.lang)} anchors. On the test scenes, it finds {_found[0]} small, {_found[1]} medium and {_found[2]} large shapes."

    explore_level = L.explore("level-slider", "level of the pyramid", _frame, start=2, values=[2, 3, 4, 5], role="",
                              task="Which level finds the small shapes, and which the large ones? Does a level ever find a shape of another size?")
    explore_level
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Every level finds the shapes of its size. Of the 248 small shapes the detector finds, level 2 finds 229 and level 3 the other 19; of the 223 large ones, level 4 finds 183, level 5 finds 36 and level 3 finds 4. Level 5 finds no box on the scene of the worked example: its largest shape, the triangle of 41 pixels, is found on level 4. The course also trained the detector without the way down, every level made only from its own map with its own 1×1 and 3×3 convolution, the *bottom-up pyramid* of the paper, which imitates SSD; and with all anchors on one map, C4 or C2:
        """),
        mo.md(C.comparison_table({"header": ("Detector", "Anchors", "Small", "Medium", "Large", "mAP", "mAP 0.75", "False boxes"),
                                  "names": {"fpn": "FPN", "bottom-up": "without the way down", "deep": "one map, C4", "sharp": "one map, C2"}}, L.lang)),
        mo.md(r"""
        Every cell gives the lowest and the highest number over the seeds 0 to 4: the small, medium and large shapes of the 200 test scenes it finds, of 268, 282 and 236; its mAP at an IoU of 0.5 and of 0.75; and its boxes of 0.5 or more that find no shape. The two pyramids find far more small shapes than C4 and give far fewer false boxes than C2. Without the way down, the pyramid does as well as the FPN on these scenes, even a little better. In the paper, on the photos of COCO, the way down made a large difference: of the small objects, the 1,000 best proposals of a pyramid without it found 30.5%, those of the FPN 44.9%, on average over IoUs from 0.5 to 0.95: their *average recall*. Exercise 5 asks why.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        A one-stage detector with a feature pyramid:

        1. The backbone makes the maps C2 to C5, each half as large as the one before.
        2. From the top down: P5 is the 1×1 convolution of C5; every level below is the 1×1 convolution of its map of the backbone, plus the level above made twice as large with `F.interpolate(..., scale_factor=2, mode="nearest")`.
        3. A 3×3 convolution on every level gives P2 to P5, all with the same number of channels.
        4. One head on every level; the rows of every level after each other, the finest first.
        5. The anchors in the same order: on level $k$, squares of $2^k$ pixels, and on every square anchors of 2 and $2\sqrt{2}$ squares wide.
        6. Train with hard negative mining: the cross-entropy over the anchors that learn a shape and the three times as many background anchors with the highest cross-entropy, plus the box loss. Detect as your one-stage detector did, with NMS for each kind.
        """),
        L.watch_out("The levels are added, not stacked: the 1×1 convolution gives every map of the backbone the same 32 channels as the level above, so that the two can be added number by number. With `torch.cat` along the channels, every level would get more channels than the one above it."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example does the steps of the recipe on a test scene with small and large shapes, `scene(26, spread=True)`, with a network that has not been trained yet; its step 6 works out what every anchor learns, the targets of training. `part2` is the backbone of your one-stage detector without its last part, which is now `part3`; `part4` and `part5` add a pooling and a convolution each. Two lists hold one convolution per level each: `lateral` the 1×1 convolutions, `smooth` the 3×3 ones. Step 2 first shows nearest-neighbor interpolation on a map of 2 by 2 numbers, then writes the way down out, level by level. In step 5, `anchor_boxes(cells, stride, sizes)` of [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) lays the anchors of one level, and `torch.zeros(n, dtype=torch.int64) + k` gives each of them the number of its level. In step 6, `match_anchors` from `everything_ai.scenes` tells which shape every anchor learns, as your `match` did: the number of the shape, or −1 or −2. Run the cell as it is.
    """)
    return


@app.cell
def example_pyramid():
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    from everything_ai.scenes import anchor_boxes, match_anchors, scene

    s = scene(26, spread=True)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255

    # 1. the backbone in four parts, each ending in a map of the pyramid; then a 1x1 and a 3x3
    #    convolution for every level, and one head
    torch.manual_seed(0)
    part2 = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    )
    part3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    part4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    part5 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
    lateral = [nn.Conv2d(c, 32, 1) for c in (32, 64, 128, 128)]
    smooth = [nn.Conv2d(32, 32, 3, padding=1) for i in range(4)]
    head = nn.Conv2d(32, 2 * 8, 3, padding=1)
    with torch.no_grad():
        c2 = part2(x)
        c3 = part3(c2)
        c4 = part4(c3)
        c5 = part5(c4)
        print(c2.shape, c3.shape, c4.shape, c5.shape)

        # 2. from the top down: nearest neighbor makes every number a block of 2 by 2; then the 1x1 map
        #    of every level plus the level above, twice as large
        small = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        print(F.interpolate(small, scale_factor=2, mode="nearest"))

        p5 = lateral[3](c5)
        p4 = lateral[2](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = lateral[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = lateral[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # 3. the 3x3 convolution of every level
        maps = [smooth[0](p2), smooth[1](p3), smooth[2](p4), smooth[3](p5)]
        print([p.shape for p in maps])

        # 4. one head on every level; the rows of every level as in your Detector, the finest first
        rows = [head(p).permute(0, 2, 3, 1).reshape(1, -1, 8) for p in maps]
        out = torch.cat(rows, dim=1)
        print([r.shape[1] for r in rows], out.shape)

    # 5. the anchors in the same order: on level k, squares of 2 ** k pixels, and on every square
    #    2 anchors, 2 and 2.83 squares wide; and the level of every anchor
    anchors, level = [], []
    for k, cells in [(2, 32), (3, 16), (4, 8), (5, 4)]:
        side = 2 ** k
        sizes = [(2 * side, 2 * side), (2 * math.sqrt(2) * side, 2 * math.sqrt(2) * side)]
        a = torch.tensor(anchor_boxes(cells, side, sizes)).float()
        anchors.append(a)
        level.append(torch.zeros(len(a), dtype=torch.int64) + k)
    anchors, level = torch.cat(anchors), torch.cat(level)
    print(anchors.shape, anchors[0], anchors[-1])

    # 6. the targets of training: which anchors learn each shape, and on which level they lie
    label = torch.tensor(match_anchors(anchors, [shape.box for shape in s.shapes]))
    for i in range(len(s.shapes)):
        top, left, bottom, right = s.shapes[i].box
        size = ((bottom - top + 1) * (right - left + 1)) ** 0.5
        print(s.shapes[i].kind, s.shapes[i].box, round(size, 1), level[label == i])
    print((label >= 0).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The four maps of the backbone have shapes (1, 32, 32, 32), (1, 64, 16, 16), (1, 128, 8, 8) and (1, 128, 4, 4): every part halves the squares each way. The map of 2 by 2 numbers becomes 4 by 4, every number a block of 2 by 2: the first row (1, 1, 2, 2). The four levels all have 32 channels, on 32 by 32 to 4 by 4 squares. The head gives 2,048, 512, 128 and 32 rows, together (1, 2720, 8), one row for each of the 2,720 anchors. The first anchor, (−2, −2, 5, 5), is 8 pixels wide around the middle of the first square of level 2; the last one, about (66.7, 66.7, 156.3, 156.3), is 90.5 pixels wide around the middle of the last square of level 5, and sticks out of the scene.

    The scene has five shapes, and 17 anchors learn one of them. The square of 10 pixels and the circle of 10.5 are learned by anchors of level 2 only, 4 and 3 of them; the circle of 20 pixels by 3 anchors of level 3; the triangle of 41 pixels by 3 of level 4. The circle of 23 pixels is learned by 3 anchors of level 3 and 1 of level 4: it lies between the sizes of the two levels, 22.6 and 32 pixels. The FPN detector has 299,728 parameters, of which the 1×1 and 3×3 convolutions of the pyramid have 48,384 and the head 4,624.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_head():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One head for every level
    """)
    return


@app.cell(hide_code=True)
def choice_head_cell():
    choice_head = L.choice(
        "head",
        question="The FPN detector has one head for all four levels: the same convolution of 3 by 3 on P2, with squares of 4 pixels, and on P5, with squares of 32. Why can the same weights find small shapes on P2 and large ones on P5?",
        options={
            "right": "On every level, the anchors are 2 and 2.83 squares wide: a shape that fits them looks alike on every level, so the head learns it once",
            "pixels": "The head works in pixels, and every level covers the same 128 by 128 pixels",
            "resized": "Before the head, every level is made 32 by 32 squares",
            "learns": "It cannot: the head learns the small shapes, and the large ones are found by the backbone",
        },
        correct="right",
        why={
            "right": "Every level has 32 channels, and its anchors are the same number of its own squares wide. A shape of 10 pixels on P2 covers as many squares as one of 80 pixels on P5, so the head can learn what a shape of about 2 squares looks like once. The paper found that a head for every level was no better: after the way down, every level knows about as much of what it sees.",
            "pixels": "The head sees squares, not pixels: 3 by 3 of them around each square. A square is 4 pixels on P2 and 32 on P5. What is the same on every level is how many squares an anchor is wide.",
            "resized": "Every level keeps its own size: P5 has 4 by 4 squares, P2 32 by 32. The head, a convolution, works on maps of any size, and gives every square its rows.",
            "learns": "The head gives every anchor of every level its logits and offsets, also the anchors of P5, which find 36 to 38 of the large shapes of the test scenes with the seeds 0 to 4. The same weights serve all levels.",
        },
    )
    choice_head
    return (choice_head,)


@app.cell(hide_code=True)
def text_exercise_pyramid():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The way down

        Complete `pyramid(cs, lateral, smooth)`, steps 2 and 3 of the recipe: `cs` is a list of the maps of the backbone, the finest first, each a tensor of shape (N, C, H, W) with half the rows and columns of the one before; `lateral` and `smooth` are lists of a 1×1 and a 3×3 convolution for every level. It gives back a list of the levels, the finest first. Start at the top, with `lateral[-1](cs[-1])`, and go down level by level, as the worked example did with `p5` to `p2`, but for any number of levels: the check tries four levels, three levels with two scenes, and the maps of the lesson's backbone. When it is right, it saves `pyramid` in your project, as `work/projects/cnn/pyramid.py`.
        """),
        L.task("Make `pyramid` give the levels of a feature pyramid network."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pyramid():
    box_pyramid = L.code_box("pyramid", lines=6)
    box_pyramid
    return (box_pyramid,)


@app.cell(hide_code=True)
def check_pyramid(box_pyramid):
    result_pyramid = L.check("pyramid", box_pyramid.value)
    result_pyramid
    return (result_pyramid,)


@app.cell(hide_code=True)
def hints_pyramid():
    L.hints(
        "pyramid",
        [
            "Step 2 of the worked example wrote the way down out for four levels, from `p5` to `p2`. Do the same in a loop over the levels: keep the levels from the top down in a list, and go from the level below the top to level 0.",
            "Start with `down = [lateral[-1](cs[-1])]` and `for i in reversed(range(len(cs) - 1)):`. Every level below is its own 1×1 map plus the last level in `down`, made twice as large: `down.append(lateral[i](cs[i]) + F.interpolate(___, scale_factor=2, mode=\"nearest\"))`.",
            "`down` now holds the top first, and the levels must come back finest first, each through its own 3×3 convolution: `return [smooth[i](down[___]) for i in range(len(cs))]`.",
        ],
        why="With four levels, the loop runs for `i` = 2, 1 and 0: `down` becomes the levels 3, 2, 1 and 0, in that order, so level `i` is `down[-1 - i]`: level 0 the last one, `down[-1]`, and level 3 the first, `down[-4]`. The 3×3 convolution comes after all the adding: what goes down is the sum, not the smoothed level.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The rows of every level

        Complete `head_rows(head, maps)`, step 4 of the recipe: `head` is one convolution, and `maps` a list of the levels, the finest first, each of shape (N, C, H, W). It gives back one tensor of shape (N, A, 8), with A the number of anchors of all levels together: on every level, the numbers that `head` gives each square, as rows of 8, as your `Detector` made them; and the rows of the levels after each other, the finest first. The check tries two levels with two scenes, three levels with one anchor per square, and the four levels of the lesson. When it is right, it saves `head_rows` in your project, as `work/projects/cnn/head_rows.py`.
        """),
        L.task("Make `head_rows` give the rows of every anchor of every level."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_head_rows = L.code_box("head-rows", lines=4)
    box_head_rows
    return (box_head_rows,)


@app.cell(hide_code=True)
def check_rows(box_head_rows):
    result_head_rows = L.check("head-rows", box_head_rows.value)
    result_head_rows
    return (result_head_rows,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "head-rows",
        [
            "Step 4 of the worked example, as a function: there, `permute` and `reshape` made the rows of every level, and `torch.cat` put them after each other.",
            "The check gives several scenes at once, so every level becomes `reshape(len(p), -1, 8)`, not `reshape(1, -1, 8)`: `len(p)` is the number of scenes, and `-1` lets `reshape` work out the number of rows.",
            "`return torch.cat([head(p).permute(___).reshape(len(p), -1, 8) for p in maps], dim=___)`.",
        ],
        why="With two scenes, a level of 8 by 8 squares and 2 anchors per square gives (2, 128, 8), and a level of 4 by 4 (2, 32, 8). Put after each other within every scene, along dimension 1, they make (2, 160, 8). Along dimension 0, they would be stacked as more scenes, and their different numbers of rows do not fit.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_background():
    mo.md(r"""
    ### Exercise 4. Which anchors count
    """)
    return


@app.cell(hide_code=True)
def choice_background_cell():
    choice_background = L.choice(
        "background",
        question="In a training scene, 10 of the 2,720 anchors learn a shape and none is ignored. How many anchors does the cross-entropy of the loss with hard negative mining count in that scene?",
        options={
            "right": "40: the 10, and the 30 background anchors with the highest cross-entropy",
            "random": "40: the 10, and 30 background anchors chosen at random",
            "all": "2,720: every anchor, as in your one-stage detector",
            "shapes": "10: only the anchors that learn a shape",
        },
        correct="right",
        why={
            "right": "Three background anchors for every anchor that learns a shape: 10 + 3 × 10 = 40. They are the ones the detector gets most wrong now, so they change from batch to batch while it learns. The other 2,680 background anchors, which it already gets right, do not count.",
            "random": "The number is right, but hard negative mining takes the background anchors with the highest cross-entropy: the hard ones. Random background anchors are mostly easy, and teach the detector almost nothing.",
            "all": "That is the loss that failed: with 2,710 background anchors against 10, the detector learns to say background everywhere, and finds 9 of the 786 shapes. Hard negative mining counts the 10 and three times as many background anchors.",
            "shapes": "Then the detector never learns background, and gives boxes everywhere. Hard negative mining also counts the 30 background anchors it gets most wrong.",
        },
    )
    choice_background
    return (choice_background,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains the course's FPN detector, the backbone of this lesson with your `pyramid` and your `head_rows`, with hard negative mining, for 8 epochs on the 2,000 training scenes with small and large shapes, in mini-batches of 32, as your one-stage detector trained. Then it finds the shapes of the 200 test scenes with your `boxes_from` and NMS, and counts what it finds of every size. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; that takes a few minutes. The chart shows the loss on the test scenes after every epoch. When it is done, you can go back through the epochs, and see the boxes the detector gave three test scenes after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train the FPN detector")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_fpn(train_button, result_pyramid, result_head_rows):
    result_pyramid, result_head_rows  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the FPN detector.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "FPN detector: the loss on the test scenes", "x": "epoch", "y": "test loss",
                                        "alt": "The loss of the FPN detector on the 200 test scenes after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**FPN detector** · epoch {_p['epoch']} of {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:
                _now["chart"].line(f"**FPN detector** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `pyramid` and your `head_rows` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than ten minutes. Does your `pyramid` or your `head_rows` do more work than it needs, such as a loop over the squares?*",
                "pyramid": "*Your `pyramid` in `work/projects/cnn/pyramid.py` stops with an error, or does not give four levels of 32 maps, from 32 by 32 to 4 by 4 squares, that training can follow back. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "head_rows": "*Your `head_rows` in `work/projects/cnn/head_rows.py` stops with an error, or does not give 2,720 rows of 8 for a scene that training can follow back. Did you change its file? Delete it, and check exercise 3 again.*",
                "offsets": "*Your `boxes_from` in `work/projects/cnn/offsets.py` stops with an error on the detector's offsets. Did you change its file? Delete it, and check its exercise in [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "The loss of the FPN detector after every epoch: on the training scenes solid, on the test scenes dashed."}
            _words = {"scene": ("scene {seed}: {n} box", "scene {seed}: {n} boxes"), "alt": "Three test scenes with small and large shapes, each with its true boxes and the dashed boxes of 0.5 or more that the detector gave it after that epoch, after NMS."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.rewind_picture(_moments[_e - 1], _words),
                        f"After epoch {_e}: test loss {L.number(_val[_e - 1], 3)}, and {_n} {'box' if _n == 1 else 'boxes'} of 0.5 or more on these three scenes, which have {C.SHOWN_SHAPES} shapes.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: which shapes get a box first, the large ones or the small ones?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _found, _shapes = _result["found"], _result["shapes"]
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("circles", "squares", "triangles"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""The FPN detector, with {C.whole(_result['params'], L.lang)} parameters, finds {sum(_found)} of the {sum(_shapes)} shapes of the 200 test scenes: a box of their kind, with a score of 0.5 or more and an IoU above 0.5. That is {_found[0]} of the {_shapes[0]} small shapes, {_found[1]} of the {_shapes[1]} medium and {_found[2]} of the {_shapes[2]} large ones. {_result['false']} of its boxes of 0.5 or more find no shape. Its mAP at an IoU of 0.5 is {L.number(_result['map'], 3)}, with an AP of {_aps}; at 0.75, it is {L.number(_result['map75'], 3)}. The course's FPN detector, trained the same way with the seeds 0 to 4, finds 727 to 741 shapes, 244 to 249 of them small, with an mAP of 0.934 to 0.950.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_top_down():
    mo.md(r"""
    ### Exercise 5. Why the way down?
    """)
    return


@app.cell(hide_code=True)
def explain_top_down_cell():
    explain_top_down = L.explain(
        "why-top-down",
        question="On the course's scenes, the pyramid without the way down finds as many small shapes as the FPN. On the photos of COCO, the way down made a large difference for small objects: an average recall of 44.9% against 30.5%. What does a sharp level need to tell what a small thing is, in a photo and in the course's scenes? And what does the way down give it?",
        model_answer="In the course's scenes, a small shape is a circle, a square or a triangle of one color on a gray background: its outline tells its kind, and every number of C2 already sees 18 by 18 pixels, as much as a small shape, so the early layers see enough. In a photo, what a small object is, a car far away or a bird, does not show in its edges and colors alone: that takes patterns that only deep layers learn, after many convolutions, and the early maps of a deep backbone know edges and colors, not things. The way down brings what the deep maps know down to the sharp levels, at their own fine squares, so the sharp levels have both: the right size, and the meaning. Without it, a sharp level has the right size for a small object, but too little meaning to tell what it is.",
        points=[
            "you say that in the course's scenes, the kind of a small shape shows in its outline and color, which the early layers already see",
            "you say that in a photo, telling what a small object is takes what deep layers learn, which the early maps do not know",
            "you say that the way down brings the meaning of the deep maps to the sharp levels, so that they have both the fine squares and the meaning",
        ],
    )
    explain_top_down
    return (explain_top_down,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The anchors, targets and boxes of [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) now work on four levels: every level its own anchors, every shape learned by the anchors whose size fits it. The 1×1 convolutions of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en) give every map of the backbone the same 32 channels, and the nearest-neighbor interpolation of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en) makes every level twice as large on the way down. Your `pyramid` builds the levels from the top down, your `head_rows` lets one head give every anchor its rows, and hard negative mining keeps the loss from drowning in background. On the course's scenes, the FPN finds 244 to 249 of the 268 small shapes, where one coarse map found 129 to 143. Your project `cnn` now has `pyramid.py` and `head_rows.py`. In the next lesson, *Focal loss and RetinaNet*, a one-stage detector with a feature pyramid gets a loss that weighs every anchor by how wrong it is, instead of choosing a few hard ones; with it, RetinaNet found objects as well as the two-stage detectors.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_head, result_pyramid, result_head_rows, choice_background, explain_top_down):
    L.wrap_up(choice_head, result_pyramid, result_head_rows, choice_background, explain_top_down)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
