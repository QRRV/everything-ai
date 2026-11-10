# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-anchors. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-anchors", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en) and [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_detr():
    quick_detr = L.choice(
        "quick-detr",
        question="DETR gives a query the box `(0.5, 0.25, 0.2, 0.1)`: `(cx, cy, w, h)`, as fractions of the photo. On a photo 600 pixels wide and 400 pixels high, where is the middle of the box, and how wide is it?",
        options={
            "right": "Middle at column 300 and row 100; 120 pixels wide",
            "height": "Middle at column 300 and row 100; 80 pixels wide",
            "swap": "Middle at column 150 and row 200; 120 pixels wide",
            "corner": "Left side at column 300 and top at row 100; 120 pixels wide",
        },
        correct="right",
        why={
            "right": "cx and w are fractions of the width: 0.5 × 600 = 300 and 0.2 × 600 = 120. cy is a fraction of the height: 0.25 × 400 = 100. DETR writes a box as its middle and its size. The detectors of this lesson make their boxes from a middle and a size too, but not from nothing.",
            "height": "w is a fraction of the width, not of the height: 0.2 × 600 = 120. See [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en).",
            "swap": "cx, the first number, is the middle across, a fraction of the width: 0.5 × 600 = 300. cy, the second, is the middle down: 0.25 × 400 = 100. See [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en).",
            "corner": "cx and cy are the middle of the box, not its top left corner: the left side lies half the width further left, at column 300 − 60 = 240. See [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en).",
        },
    )
    quick_detr
    return (quick_detr,)


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="A CNN halves the height and the width of its maps three times, each time with stride 2. For a scene of 128 by 128 pixels, how large are its last feature maps, and how many pixels of the scene does each square of them stand for?",
        options={
            "right": "16 by 16; each square stands for 8 by 8 pixels",
            "once": "64 by 64; each square stands for 2 by 2 pixels",
            "sixteen": "16 by 16; each square stands for 16 by 16 pixels",
            "four": "8 by 8; each square stands for 16 by 16 pixels",
        },
        correct="right",
        why={
            "right": "128 / 2 / 2 / 2 = 16, and 128 / 16 = 8: each square of a map stands for 8 by 8 pixels of the scene. Today every one of those 256 squares gets boxes of its own.",
            "once": "That halves once. Three halvings go 128, 64, 32, 16. See [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).",
            "sixteen": "16 squares across 128 pixels is 128 / 16 = 8 pixels per square, not 16. See [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).",
            "four": "That halves four times: 128, 64, 32, 16, 8. Three times gives 16 by 16. See [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    DETR, in [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en), learned to give each of its 100 object queries a box from nothing: anywhere in the photo, of any size. Many detectors before it, and many since, make that easier. They lay a fixed set of boxes over the whole photo, a few sizes on every square of a feature map, and learn for each box only two things: whether it lies on an object, and of which kind; and how far it must move and stretch to fit. Those fixed boxes are called anchors. Faster R-CNN (Ren and colleagues, 2015) introduced them, and SSD, YOLOv2 and RetinaNet use them too. One hard question, where are the boxes, becomes many small ones, each about one anchor near an object. But to train them, every anchor needs a right answer: which object, if any, should it learn? The IoU of [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en) decides that.

    **An everyday example.** A clothes shop does not sew a coat from nothing for every customer. It has a few sizes on the rack. You take the one that fits best, and the tailor only takes it in or lets it out a little: a small change from a size that was already close. How well each size fits decides where the tailor starts.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "how-many",
        question="The scene of the worked example has 4 shapes. A feature map of 16 by 16 squares gets 3 anchors per square, of 16, 24 and 32 pixels: 768 anchors. An anchor learns a shape when its IoU with that shape is 0.5 or more, or when it is that shape's best anchor. How many of the 768 learn a shape?",
        options={
            "four": "4: one per shape",
            "right": "20",
            "hundred": "About 100",
            "half": "About 400: half of them",
        },
        reveal="**20.** 8 anchors learn the circle, 6 the square, 4 the large triangle and 2 the small triangle. 735 learn background, and 13 are ignored. Over 500 scenes, it is 17.2 anchors per scene on average: 2.2% of the anchors. Almost every anchor learns that there is nothing there.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Anchors on every square.** A CNN for object detection turns the scene into feature maps, as the CNNs of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en) do. With three halvings, a scene of 128 by 128 pixels gives maps of 16 by 16 squares, each standing for 8 by 8 pixels. An **anchor box**, anchor for short, is a fixed box around the middle of such a square. Every square gets the same few anchors, of a few sizes and shapes. Faster R-CNN takes 3 sizes, with areas of 128², 256² and 512² pixels, and 3 shapes, 1:1, 1:2 and 2:1: 9 anchors per square, "roughly 20000" on a photo of 1000 by 600 pixels. The boxes of the course's shapes are 9 to 44 pixels on a side, and almost square: for 90% of them, the height divided by the width lies between 0.88 and 1.13. So the course takes 3 square sizes, 16, 24 and 32 pixels: on 16 by 16 squares, 768 anchors.

        **What each anchor learns.** For every anchor, the detector gives a kind, one of the shapes or background, and four numbers that move and stretch the anchor into a box. To train it, every anchor needs a right answer, and its IoU with the true boxes decides it. Each anchor takes the true box it overlaps most. With an IoU of 0.5 or more, it learns that box: its kind, and the four numbers that turn the anchor into it. Below 0.4, it learns background. In between, it is ignored: it lies partly on a shape, and either answer would teach the detector something partly wrong. These are the thresholds of RetinaNet (Lin and colleagues, 2017); Faster R-CNN takes 0.7 and 0.3, and SSD 0.5. Last, every true box also gets its best anchor, the anchor with its highest IoU, even below 0.5, and even when that anchor overlaps another box more, so that no object is left without one. Faster R-CNN adds this rule because "in some rare cases" the threshold alone "may find no positive sample": no anchor that learns the box. On the course's scenes, 25 of the 2012 shapes of 500 scenes need it; their best anchors reach 0.40 or more. When the middle of a box lies halfway between two squares, two anchors can share the highest IoU; Faster R-CNN's rule names "the anchor/anchors" with the highest IoU, so both learn the box.

        **Box offsets.** An anchor does not learn the four sides of its box, but how far its box is off: the shift of the middle, as a share of the anchor's height and width, and how many times larger the box is, as a log. These four numbers are the **box offset** of the anchor, as Faster R-CNN writes them, here rows first:
        """),
        L.formula(r"t_y = \frac{y - y_a}{h_a}, \qquad t_x = \frac{x - x_a}{w_a}, \qquad t_h = \log\frac{h}{h_a}, \qquad t_w = \log\frac{w}{w_a}"),
        mo.md(r"""
        Here $(y, x)$ is the middle of the true box and $h$ and $w$ are its height and width; $y_a$, $x_a$, $h_a$ and $w_a$ are the same for the anchor. A box of the course holds both ends, so its height is bottom − top + 1, and its middle (top + bottom) / 2. An anchor exactly on its box has four zeros. Dividing by the anchor's size makes the numbers alike for small and large boxes: a shift of half the anchor is 0.5 for both. The log makes twice as large and half as large equally far from 0, 0.69 and −0.69, and the way back can never give a negative size:
        """),
        L.formula(r"y = y_a + t_y \, h_a, \qquad h = h_a \, e^{t_h}"),
        mo.md(r"""
        and the same for $x$ and $w$; from the middle and the size come the sides, top = y − (h − 1) / 2 and bottom = y + (h − 1) / 2. Because an anchor learns only boxes it already overlaps well, almost always by 0.5 or more, its offsets stay small: over 500 scenes, a shift is on average 0.11 of the anchor's size, up or down, left or right, and a log of the size 0.15, larger or smaller; 95% stay below 0.25 and 0.32.
        """),
        L.watch_out("An anchor learns at most one box: normally the one it overlaps most, but the last rule can give it another box, one whose best anchor it is. A box can have many anchors: the circle has 8. An anchor gives one kind and one box offset, so it cannot learn two boxes, even when it overlaps both by 0.5 or more."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **anchor box** (anchor) | a fixed box around the middle of a square of a feature map; every square has the same few sizes and shapes, and the detector learns for each anchor a kind and a box offset | `anchors[i]` |
        | **box offset** | the four numbers that turn an anchor into a box: the shift of the middle as a share of the anchor's height and width, and the log of how many times higher and wider the box is | `t` |
        | $y_a$, $x_a$, $h_a$, $w_a$ | the middle and the size of an anchor: its row and column, its height and width | `middle_a`, `size_a` |
        | $t_y$, $t_x$, $t_h$, $t_w$ | the box offset: the shift down and across, the change of height and of width | `t[:, :2]`, `t[:, 2:]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_anchors():
    mo.vstack([
        mo.md("**Anchors on the scene of the worked example**"),
        mo.Html(C.anchors_picture({
            "one": "the {n} anchors of one square",
            "learn": "the {n} anchors that learn a shape",
            "alt": "Two pictures of a scene with two triangles, a blue square and a circle. Left, the squares of the feature map as thin lines, one square darker, with three dashed boxes of 16, 24 and 32 pixels around its middle, and the circle's true box, solid. Right, the four true boxes, solid, and the 20 dashed anchors that learn a shape: 8 around the circle, 6 around the blue square, 4 around the large triangle and 2 at the small triangle.",
        })),
        mo.md(r"""
        Left, the square of the map in row 9 and column 8, square (9, 8), with its three anchors and the circle's true box: the anchor of 32 pixels fits it with an IoU of 0.64. Right, the 20 anchors that learn a shape. The small triangle has no anchor with an IoU of 0.5 or more: its two best, of 16 pixels, reach 0.49 each, and learn it through the last rule.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Do you need more than one size? The slider gives every square of the map one anchor, as high as it is wide: 8 to 56 pixels. On the left, the anchors that reach an IoU of 0.5 with a shape of the scene; on the right, for every size, the share of the shapes of 500 scenes that have such an anchor. The dotted line is the course's three sizes together.
    """)
    return


@app.cell(hide_code=True)
def explore_size_cell():
    def _frame(z):
        z = int(z)
        _html, _stop = C.slider_frame(z, {"title": "anchors of {size} pixels: {found} of {n} shapes", "three": "16, 24 and 32: {share}%",
                                          "size": "size of the anchor in pixels", "share": "shapes of 500 scenes with an anchor\nwhose IoU is 0.5 or more", "point": ".",
                                          "alt": "The scene with its four true boxes and the dashed anchors of {size} by {size} pixels that reach an IoU of 0.5 with a shape: {found} of the {n} shapes have one. Next to it, the share of the shapes of 500 scenes with such an anchor at every size from 8 to 56 pixels: 0 at 8, rising to 81% at 28, and falling to 7% at 56, always below the dotted line of the three sizes together, at 98.8%. At {size} pixels it is {share}%."})
        return _html, f"Anchors of {z} by {z} pixels: {len(_stop['positive'])} with an IoU of 0.5 or more, for {_stop['found']} of the 4 shapes. Over 500 scenes, {L.number(100 * C.curve()[z], 1)}% of the shapes get one."

    explore_size = L.explore("size-slider", "size of the anchor in pixels", _frame, start=16, values=C.SLIDER, role="hyper",
                             task="Find the size at which the most shapes of the 500 scenes get an anchor. Do all four shapes of this scene get one there?")
    explore_size
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    28 pixels gives the most: 81.0% of the shapes of 500 scenes, and 3 of the 4 shapes here, but not the small triangle. It gets an anchor with an IoU of 0.5 or more only at 18 and 20 pixels, and the blue square only from 24 to 46: no one size fits shapes of 9 to 44 pixels. The three sizes together give 98.8%; with Faster R-CNN's 0.7, only 55.0% would. More anchor shapes do not help here: 3 sizes times Faster R-CNN's 3 shapes, 9 anchors per square, give 99.0%, because the boxes are almost square. Faster R-CNN saw the same on PASCAL VOC: one anchor per square scored 3 to 4 points of mAP less than its nine, and 3 sizes of one shape, at 69.8% mAP, did as well as 3 sizes of 3 shapes, at 69.9%. And the squares must lie close enough together: with the same three sizes on 8 by 8 squares of 16 pixels, only 69.6% of the shapes get an anchor with an IoU of 0.5 or more, because the middle of a small shape can then lie far from every anchor.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        To tell every anchor what to learn:

        1. Lay the anchors: around the middle of every square of the feature map, one anchor of each size and shape. With squares of $s$ pixels, the middle of square $(i, j)$ lies at row $i \cdot s + (s - 1)/2$ and column $j \cdot s + (s - 1)/2$.
        2. Compute the IoU of every anchor with every true box: an array with a row per anchor and a column per box.
        3. Match. Each anchor takes the box it overlaps most: from 0.5 on it learns that box, from 0.4 up to 0.5 it is ignored, below 0.4 it learns background. Then every box gets its best anchors, those with its highest IoU, even below 0.5 and even when they overlap another box more.
        4. For every anchor that learns a box, compute its box offset: $t_y = (y - y_a)/h_a$, $t_x = (x - x_a)/w_a$, $t_h = \log(h/h_a)$, $t_w = \log(w/w_a)$, with the middle (top + bottom) / 2 and the height bottom − top + 1.
        5. The way back, for the offsets a detector predicts: $y = y_a + t_y h_a$ and $h = h_a e^{t_h}$, and the same across; then the sides, top = y − (h − 1) / 2 and bottom = y + (h − 1) / 2.
        """),
        L.watch_out("Give every box its best anchors last. When the thresholds come after that rule, a best anchor between 0.4 and 0.5 is ignored again, and a box like the small triangle is left without an anchor."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe on the scene of [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), made again with `scene(18)`. The first cell does steps 1 to 3 in NumPy. `iou_matrix` is the IoU of [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en) for all anchors at once: `anchors[:, 0]` is the column with all their tops, and `np.minimum` takes the smaller of two numbers, element by element, as `np.maximum` takes the larger. Step 3 uses boolean masks: `most >= 0.5` is `True` for every anchor whose largest IoU is 0.5 or more, and `label[...] = ...` sets only those. The second cell does steps 4 and 5 for the circle's best anchor. Run them as they are; then change `sizes` where the comment says, to add a tall and a wide anchor, and run them again.
    """)
    return


@app.cell
def example_match():
    import numpy as np
    from everything_ai.scenes import scene

    s = scene(18)
    truth = np.array([shape.box for shape in s.shapes], dtype=float)   # the true boxes, one row per shape

    # 1. the anchors: around the middle of each square of a map of 16 by 16 squares,
    #    each square 8 by 8 pixels of the scene, one anchor of each size
    cells, stride = 16, 8
    sizes = [(16, 16), (24, 24), (32, 32)]   # (height, width); change: add (34, 17) and (17, 34)
    rows = []
    for i in range(cells):
        for j in range(cells):
            y = i * stride + (stride - 1) / 2   # the middle of square (i, j)
            x = j * stride + (stride - 1) / 2
            for h, w in sizes:
                rows.append((y - (h - 1) / 2, x - (w - 1) / 2, y + (h - 1) / 2, x + (w - 1) / 2))
    anchors = np.array(rows)
    print(truth)
    print(anchors.shape)
    print(anchors[:3])


    def iou_matrix(anchors, boxes):
        # the IoU of every anchor with every box, as in cv-iou, for all anchors at once
        ious = np.zeros((len(anchors), len(boxes)))
        for k in range(len(boxes)):
            box = boxes[k]
            top = np.maximum(anchors[:, 0], box[0])
            left = np.maximum(anchors[:, 1], box[1])
            bottom = np.minimum(anchors[:, 2], box[2])
            right = np.minimum(anchors[:, 3], box[3])
            overlap = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)
            area_a = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
            area_b = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            ious[:, k] = overlap / (area_a + area_b - overlap)
        return ious


    # 2. the IoU of every anchor with every true box: one column per box
    ious = iou_matrix(anchors, truth)
    print(ious.shape, ious.max(axis=0))
    # 3. match: each anchor takes the box it overlaps most; from 0.5 on it learns
    #    that box, below 0.4 background (-1), in between it is ignored (-2)
    best = ious.argmax(axis=1)
    most = ious.max(axis=1)
    label = np.zeros(len(anchors), dtype=int) - 1
    label[most >= 0.5] = best[most >= 0.5]
    label[(most >= 0.4) & (most < 0.5)] = -2
    # and every true box also gets its best anchors, even below 0.5
    for k in range(len(truth)):
        label[ious[:, k] == ious[:, k].max()] = k
    print((label >= 0).sum(), (label == -2).sum(), (label == -1).sum())
    for k in range(len(truth)):
        print(s.shapes[k].kind, (label == k).sum())
    return anchors, ious, label, np, truth


@app.cell
def example_offsets(anchors, ious, np, truth):
    # 4. the offsets of the circle's best anchor: how far the middle moves, as a share
    #    of the anchor's height and width, and the log of how much larger the box is
    anchor = anchors[ious[:, 3].argmax()]
    box = truth[3]
    ya, xa = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    ha, wa = anchor[2] - anchor[0] + 1, anchor[3] - anchor[1] + 1
    yb, xb = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hb, wb = box[2] - box[0] + 1, box[3] - box[1] + 1
    t = np.array([(yb - ya) / ha, (xb - xa) / wa, np.log(hb / ha), np.log(wb / wa)])
    print(anchor, box)
    print(t)
    # 5. back: the offsets turn the anchor into the true box again
    y2, x2 = ya + t[0] * ha, xa + t[1] * wa
    h2, w2 = ha * np.exp(t[2]), wa * np.exp(t[3])
    print(np.array([y2 - (h2 - 1) / 2, x2 - (w2 - 1) / 2, y2 + (h2 - 1) / 2, x2 + (w2 - 1) / 2]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `truth` holds the four true boxes, and `anchors` has shape (768, 4): 256 squares times 3 sizes. The first three are the anchors of square (0, 0), whose middle lies at row and column 3.5; they stick out of the scene, and that is fine. `ious` has shape (768, 4). The highest IoU of each shape is 0.49 for the small triangle, 0.67 for the large triangle, 0.70 for the blue square and 0.64 for the circle. 20 anchors learn a shape, 13 are ignored, and 735 learn background. The small triangle has 2 anchors, though neither reaches 0.5: its middle lies at column 15.5, halfway between the middles of the squares in columns 1 and 2, at 11.5 and 19.5, so its two best anchors both reach 0.4884, and the last rule gives it both.

    In the second cell, the circle's best anchor is (60, 52, 91, 83), 32 by 32 pixels around the middle of square (9, 8). The circle's box, (58, 50, 86, 78), has its middle 3.5 pixels higher and 3.5 pixels further left, and is 29 pixels high and wide. So the offsets are −3.5 / 32 = −0.109, twice, and log(29 / 32) = −0.098, twice. Step 5 turns the anchor back into the circle's box, exactly. With the tall and the wide anchor added, there are 1280 anchors: 27 learn a shape, 28 are ignored and 1225 learn background. No shape gets a better best anchor from them, because its box is almost square.
    """)
    return


@app.cell(hide_code=True)
def text_torch_step():
    mo.md(r"""
    ### One rung up: the same in PyTorch

    A detector gives its offsets as tensors, so here is step 4 in PyTorch, for all 20 anchors that learn a box at once. `anchors[learns]` keeps the rows of the anchors that learn a box, and `truth[label[learns]]` the true box of each: an array as an index. `anchors_t[:, :2]` is the pair (top, left) of every anchor and `anchors_t[:, 2:]` the pair (bottom, right), so one line gives the middles of all of them, and one their sizes. `torch.log` works as `np.log`, and `torch.cat(..., dim=1)` puts the shift and the change of size side by side.

    | NumPy, one anchor | PyTorch, all anchors at once |
    |---|---|
    | `ya, xa = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2` | `middle_a = (anchors_t[:, :2] + anchors_t[:, 2:]) / 2` |
    | `ha, wa = anchor[2] - anchor[0] + 1, anchor[3] - anchor[1] + 1` | `size_a = anchors_t[:, 2:] - anchors_t[:, :2] + 1` |
    | `np.array([(yb - ya) / ha, (xb - xa) / wa, np.log(hb / ha), np.log(wb / wa)])` | `torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)` |
    """)
    return


@app.cell
def example_torch(anchors, label, truth):
    import torch

    # 4. the offsets of every anchor that learns a box at once, in PyTorch
    learns = label >= 0
    anchors_t = torch.tensor(anchors[learns]).float()       # shape (20, 4)
    boxes_t = torch.tensor(truth[label[learns]]).float()    # the box each of them learns
    middle_a = (anchors_t[:, :2] + anchors_t[:, 2:]) / 2   # (y, x) of each anchor
    size_a = anchors_t[:, 2:] - anchors_t[:, :2] + 1       # (h, w) of each anchor
    middle = (boxes_t[:, :2] + boxes_t[:, 2:]) / 2
    size = boxes_t[:, 2:] - boxes_t[:, :2] + 1
    targets = torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)
    print(targets.shape)
    print(targets[-3:])
    print(targets.abs().max())
    return


@app.cell(hide_code=True)
def text_after_torch():
    mo.md(r"""
    `targets` has shape (20, 4): a box offset for every anchor that learns a box, what a detector must learn to predict. The first of the last three rows is the circle's best anchor, as in the NumPy cell. The other two are the small triangle's anchors. For both, the triangle's middle lies 0.125 of the anchor's height lower; for one it lies 0.25 of its width to the right, for the other 0.25 to the left. Both anchors have the size of the triangle's box, 16 by 16, so their changes of size are 0. The largest offset in the scene is 0.35.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_learn():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What each anchor learns
    """)
    return


@app.cell(hide_code=True)
def choice_learn_cell():
    choice_learn = L.choice(
        "learn",
        question="Four anchors and two true boxes, A and B. Anchor 1 has an IoU of 0.62 with A and 0.30 with B; anchor 2 has 0.45 with A and 0.05 with B; anchor 3 has 0.15 with A and 0.42 with B; anchor 4 has 0.10 with A and 0.20 with B. No other anchor overlaps B more than anchor 3. What does each anchor learn?",
        options={
            "right": "1 learns A, 2 is ignored, 3 learns B, 4 learns background",
            "no-rule": "1 learns A, 2 is ignored, 3 is ignored, 4 learns background",
            "low": "1 learns A, 2 learns A, 3 learns B, 4 learns background",
            "no-band": "1 learns A, 2 learns background, 3 learns B, 4 learns background",
        },
        correct="right",
        why={
            "right": "Anchor 1 reaches 0.5 with A. Anchor 2's best is 0.45, between 0.4 and 0.5: ignored. Anchor 3's best, 0.42, lies in between too, but it is B's best anchor, so the last rule gives it B; without that rule, nothing would learn B. Anchor 4 stays below 0.4: background.",
            "no-rule": "B's anchors all stay below 0.5, so by the thresholds alone, nothing learns B. The last rule gives every box its best anchor, even below 0.5: anchor 3.",
            "low": "An anchor learns a box from 0.5 on. Anchor 2, at 0.45, lies partly on A: it is ignored, so that it learns neither A nor background.",
            "no-band": "Below 0.4 is background; from 0.4 up to 0.5 an anchor is ignored. Anchor 2, at 0.45, lies partly on A: if it learned background, it would teach the detector that part of A is nothing.",
        },
    )
    choice_learn
    return (choice_learn,)


@app.cell(hide_code=True)
def text_exercise_match():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Match the anchors

        Complete `match(ious)`: it gets the IoU of every anchor with every true box, an array with a row per anchor and a column per box, and gives back an array with one whole number per anchor: the number of the box it learns, -1 for background, -2 when it is ignored. `iou_matrix` from the worked example is given. The check gives your function the four anchors of exercise 1, IoUs of exactly 0.5 and 0.4, the scene of the worked example, and another scene with other anchors. For the scenes, it lays the anchors with `anchor_boxes` from `everything_ai.scenes`, which does what step 1 of the worked example does: `anchor_boxes()` gives the same 768 anchors, and `anchor_boxes(cells=8, stride=16, sizes=((24, 24), (40, 40)))` two sizes on 8 by 8 squares of 16 pixels. When it is right, it saves `match` and `iou_matrix` in your project, as `work/projects/cnn/match.py`.
        """),
        L.task("Make `match` give every anchor what it learns: a box, background, or nothing."),
    ])
    return


@app.cell(hide_code=True)
def exercise_match():
    box_match = L.code_box("match", lines=10)
    box_match
    return (box_match,)


@app.cell(hide_code=True)
def check_match(box_match):
    result_match = L.check("match", box_match.value)
    result_match
    return (result_match,)


@app.cell(hide_code=True)
def hints_match():
    L.hints(
        "match",
        [
            "Step 3 of the recipe. For every anchor, its row of `ious` holds its IoU with each box: `ious.argmax(axis=1)` gives the box it overlaps most, and `ious.max(axis=1)` that IoU.",
            "Start every anchor at -1, then use boolean masks, as the worked example does: `label[most >= 0.5] = best[___]`, and -2 where `most` is from 0.4 up to 0.5.",
            "Last, go through the columns: `for k in range(ious.shape[1]):`, and give every anchor with the highest IoU of column `k` the number `k`: `label[ious[:, k] == ___] = k`.",
        ],
        why="For exercise 1: anchor 1 learns A, 0, at 0.62; anchor 2 is ignored, -2, at 0.45; anchor 4 learns background, -1, at 0.20; and anchor 3, at 0.42, is first ignored, then B's best anchor, so it learns B, 1. `match` gives `[0, -2, 1, -1]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_boxes_from():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. From offsets back to boxes

        A detector gives a box offset for every anchor, and you need its boxes. Complete `boxes_from(anchors, t)`: it gets anchors, a tensor of shape (N, 4) with rows `(top, left, bottom, right)`, and their offsets, a tensor of shape (N, 4) with rows $(t_y, t_x, t_h, t_w)$, and gives back the boxes, a tensor of shape (N, 4): step 5 of the recipe, for all anchors at once, in PyTorch. `offsets`, the PyTorch cell of the worked example as a function, is given: `boxes_from` is its way back. The check starts with the circle's best anchor and its offsets, then tries offsets of 0, a box twice as high and half as wide, and the way back from `offsets`. When it is right, it saves `boxes_from` and `offsets` in your project, as `work/projects/cnn/offsets.py`.
        """),
        L.task("Make `boxes_from` turn anchors and their box offsets into boxes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_boxes_from():
    box_boxes_from = L.code_box("boxes-from", lines=8)
    box_boxes_from
    return (box_boxes_from,)


@app.cell(hide_code=True)
def check_boxes_from(box_boxes_from):
    result_boxes_from = L.check("boxes-from", box_boxes_from.value)
    result_boxes_from
    return (result_boxes_from,)


@app.cell(hide_code=True)
def hints_boxes_from():
    L.hints(
        "boxes-from",
        [
            "Start as `offsets` does: the middle and the size of every anchor, `middle_a` and `size_a`, each of shape (N, 2). Then step 5: the middle of the box is the anchor's middle plus `t[:, :2]` times the anchor's size.",
            "`t[:, 2:]` holds logs: undo them, so that the size of the box is `size_a * torch.___(t[:, 2:])`.",
            "From the middle and the size come (top, left) and (bottom, right), both ends included: `torch.cat([middle - ___, middle + ___], dim=1)`.",
        ],
        why="For the circle's anchor, (60, 52, 91, 83): its middle is (75.5, 67.5) and its size 32 by 32. The shift −0.1094 × 32 = −3.5 moves the middle to (72, 64), and 32 × e^−0.0984 = 29 is the size. The sides are 72 − 14 = 58 and 72 + 14 = 86, and across 50 and 78: the circle's box.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 4. Why offsets?
    """)
    return


@app.cell(hide_code=True)
def explain_why_cell():
    explain_why = L.explain(
        "why-offsets",
        question="A detector could also learn the four sides of every box in pixels, (top, left, bottom, right), on every square. Why does it learn box offsets from an anchor instead?",
        model_answer="**The same object gives the same offsets wherever it lies.** A convolution slides the same filter over every square, so a CNN gives the same numbers for the same pattern anywhere; offsets from the square's own anchor are numbers it can give. Sides in pixels would need other numbers for the same circle at another place. **The numbers are small, and alike for every size.** An anchor learns only boxes it already overlaps well, almost always by 0.5 or more, so its offsets stay near 0: over 500 scenes, 95% stay below 0.25 and 0.32. Divided by the anchor's size, a shift of half the anchor is 0.5 for a small and a large anchor alike, so the loss weighs every object alike. **Each anchor has an easier task**: only the objects close to its square, of about its size.",
        points=[
            "you say that the same object gives the same offsets on every square, where sides in pixels depend on the place, and that a convolution does the same everywhere",
            "you say that the offsets are small numbers around 0, alike for small and large objects, because they are shares of the anchor's size",
            "you say that each anchor only has to handle objects close to it, of about its size",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The IoU of [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en) now decides what every anchor learns: from 0.5 on the box it overlaps most, below 0.4 background, in between nothing, and every box its best anchors. The feature maps of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en) give the squares. Where DETR, in [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en), learned every box from nothing, an anchor learns a box offset: a shift as a share of its size, and the log of its change of size. Almost every anchor learns background: 2.2% of them learn a shape. Your project `cnn` now has `match.py` and `offsets.py`. In the next lesson, *Project: your own one-stage detector*, a CNN gives a kind and a box offset for every anchor at once, and you train it on the course's scenes with these targets.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_learn, result_match, result_boxes_from, explain_why):
    L.wrap_up(choice_learn, result_match, result_boxes_from, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
