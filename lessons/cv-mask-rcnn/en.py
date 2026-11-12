# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-mask-rcnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-mask-rcnn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en) and [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_round():
    quick_round = L.choice(
        "quick-round",
        question="RoI pooling took a proposal from row 41.4 to row 67.7 from the squares 5 to 8 of the feature maps, every square 8 by 8 pixels. The same proposal 3 pixels lower runs from row 44.4 to 70.7. What does RoI pooling give it?",
        options={
            "right": "Exactly the same numbers: it still takes the squares 5 to 8",
            "next": "The numbers of the squares 6 to 9: 3 pixels is enough for the next square",
            "little": "Almost the same numbers: every bin moves 3/8 of a square down",
        },
        correct="right",
        why={
            "right": "`int(44.4 // 8)` is 5 and `int(70.7 // 8)` is 8, as before: the same squares, the same bins, the same 1,024 numbers. The second stage cannot see that the proposal moved. Today's RoIAlign does not round.",
            "next": "`44.4 // 8` is 5 and `70.7 // 8` is 8: both corners stay in their squares. See [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en).",
            "little": "RoI pooling cuts the squares, not the proposal, into bins. The squares stay 5 to 8, so the numbers stay exactly the same. See [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en).",
        },
    )
    quick_round
    return (quick_round,)


@app.cell(hide_code=True)
def quick_check_merge():
    quick_merge = L.choice(
        "quick-merge",
        question="In a scene, two circles touch. What does their class map give them, as your FCN learned to make it?",
        options={
            "right": "One area of the class circle: nothing in it says where one circle stops",
            "two": "Two areas of the class circle, with a line of background between them",
            "new": "The class circle for one and another class for the other",
        },
        correct="right",
        why={
            "right": "A class map gives every pixel a kind, and both circles are of the same kind: their pixels are one area of the class circle. How many circles there are, and which pixel belongs to which, is not in it.",
            "two": "Every pixel of a circle gets the class circle, also where the circles touch: there is no pixel of background between them. See [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).",
            "new": "A class map has one class for every kind: background, circle, square and triangle. Both circles get the class circle. See [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).",
        },
    )
    quick_merge
    return (quick_merge,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your FCN gives every pixel a kind. When two circles touch, its class map makes one area of them, and nothing says where one circle stops and the other begins. That is not rare: of the 808 shapes of the course's 200 test scenes, 186 touch or overlap a shape of the same kind, in 82 of the scenes. Your detectors give every object a box of its own, but a box is not the object: of the pixels in the box of a circle, a median of 78% are circle; of a square, which may be turned, 59%; of a triangle, 50%. **Instance segmentation** gives every object a mask of its own: the pixels that are that object. Mask R-CNN (He, Gkioxari, Dollár and Girshick, 2017) does it with a small addition to Faster R-CNN: for every box it finds, a small FCN draws the mask of the object inside. Its makers write that it "adds only a small overhead to Faster R-CNN, running at 5 fps": five pictures a second on a GPU. On COCO, the masks of its best model had a mask AP of 37.1, against 33.6 for FCIS+++, the winner of the segmentation challenge of COCO in 2016. Today you build it on the course's scenes, on top of your two-stage detector of [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en).
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_align():
    guess_align = L.guess(
        "align",
        question="Faster R-CNN cuts every proposal from the feature maps with RoI pooling, which rounds to whole squares. The makers of Mask R-CNN replaced it with RoIAlign, which does not round, and changed nothing else. With a ResNet-50 backbone, whose squares are 16 by 16 pixels, how much better did the masks get on COCO, in mask AP?",
        options={
            "hardly": "Hardly: about 0.3 points",
            "right": "About 3 points: from 26.9 to 30.3",
            "lots": "About 15 points: from 26.9 to 42",
            "worse": "They got worse: rounding helps against noise",
        },
        reveal="About 3 points: from 26.9 to 30.3 mask AP, the mean over COCO's ten IoU thresholds. At an IoU of 0.5, from 48.8 to 51.2; at 0.75, from 26.4 to 31.5: the stricter the threshold, the larger the gain. With squares of 32 by 32 pixels, rounding costs more: there RoIAlign gained 7.3 points of mask AP, and 10.5 at 0.75. Its makers sum it up as 10% to 50% better masks, relatively.",
    )
    guess_align
    return (guess_align,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A mask for every object.** A class map gives every pixel a kind. **Instance segmentation** gives every object, here every shape, a mask of its own: the pixels that are that object, also when it touches another of its kind. **Mask R-CNN** takes the two-stage detector of [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en) and gives it a third head, next to the second stage: the **mask head**. The second stage gives every proposal a kind and a better box; the mask head draws a mask in it.

        **The mask head is a small FCN.** RoIAlign, below, cuts a box to 8 by 8 bins: 64 × 8 × 8 numbers. Two convolutions of 3 by 3 with 32 channels and ReLU, a transposed convolution of 2 by 2 with stride 2 to 16 by 16, ReLU, and a convolution of 1 by 1 to 3 maps of 16 by 16 logits: a mask for every kind. Its 31,939 parameters make the course's two-stage detector of 167,386 parameters a Mask R-CNN of 199,325. In the paper, the head of its best models is built the same way: four convolutions of 3 by 3, a transposed convolution of 2 by 2 with stride 2 and a convolution of 1 by 1, which draw masks of 28 by 28, one for each of COCO's 80 kinds. The ResNet-50 of its comparisons, such as the one in the guess, draws masks of 14 by 14.

        **Only the mask of its kind learns.** The second stage decides the kind of a box; the mask head draws a mask for every kind, and only the mask of the box's kind counts. The sigmoid of every logit is the probability that its pixel belongs to the object, and the loss is the mean log loss of the 256 pixels, as in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en). The masks of the kinds do not compete for a pixel: the paper calls this decoupling the mask from the class. With softmax over the kinds at every pixel, as in a class map, its masks were worse: a mask AP of 24.8 instead of 30.3.

        **What a mask learns.** In training, every proposal that learns a kind, with an IoU of 0.5 or more with a true box, also learns a mask: the true mask of that box's shape, cut to the proposal. For each of the 16 by 16 bins of the proposal, the target is 1 when the pixel that the bin's middle lies in belongs to the shape, and 0 when not. A proposal that is a little too large thus learns a circle that does not reach its edges. A mask of 16 by 16 holds most of a shape: the true masks of the test scenes, cut to 16 by 16 bins of their own box this way and made as large as the box again, have a median IoU of 0.941 with the originals; with 8 by 8, 0.882; with 14 by 14, the size of the paper's comparisons, 0.930; with 28 by 28, that of its best models, 0.993. The network trains on the loss of the RPN plus the loss of the second stage plus the loss of the mask head.

        **RoIAlign: no rounding.** Square s of the feature maps covers the pixels 8s to 8s + 7, so its middle lies at pixel 8s + 3.5, and pixel y lies at (y − 3.5) / 8 squares. **RoIAlign** cuts the box itself, not the squares it lies in, into k by k bins of the same size. It takes the middle of every bin, in pixels and then in squares, without rounding, and the value of the feature maps there, mixed from the four squares around it with bilinear interpolation, as your `value_at` of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en) mixed four pixels. A box from row 58 to row 86 covers 29 pixels, from 57.5 to 86.5; the middle of the first of its 8 bins lies at 57.5 + 29 / 16 = 59.3125, square 6.9766. Training follows the loss back through the mix into the four squares. The paper takes four points in every bin and their mean or their maximum, and found that the result is "not sensitive to the exact sampling locations, or how many points are sampled"; here, the course takes one point per bin. The second stage gets RoIAlign too, with 4 by 4 bins.

        **Pasting a mask.** To detect, the mask head draws a mask for every box that is left after NMS, and the mask of the box's kind is its mask. Its 16 by 16 probabilities are made as large as the box with bilinear interpolation, and every pixel of 0.5 or more belongs to the object: the paper's mask is "resized to the RoI size, and binarized at a threshold of 0.5". That is `paste`. The paper draws masks for the 100 boxes with the highest score.

        **Mask AP.** As the mAP of [Score a detector: mean average precision](/course/open/cv-map?lang=en), with the IoU of masks instead of boxes: a mask finds a shape of its kind when their IoU is above the threshold. COCO's mask AP, the number in the paper, is the mean at its ten thresholds, 0.5, 0.55 and so on up to 0.95.
        """),
        L.watch_out("The mask head never decides the kind of a box. A circle that the second stage calls a square gets the mask the head draws for squares."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **instance segmentation** | a mask for every object: the pixels that are that object, also when it touches another object of its kind | |
        | **Mask R-CNN** | Faster R-CNN with a mask head, by He, Gkioxari, Dollár and Girshick (2017) | `MaskRCNN` |
        | **mask head** | a small FCN that draws, in every box, a mask of logits for every kind; only the mask of the box's kind counts | `self.mask_head` |
        | **RoIAlign** | cutting the features of a box to k by k bins without rounding: the value of the feature maps at the middle of every bin, with bilinear interpolation | `roi_align` |
        | **mask AP** | the AP of masks: as mAP, with the IoU of masks instead of boxes | |
        | `maps[:, rows][:, :, cols]` | with lists of places: the rows `rows` of every channel, then the columns `cols` of those; every pair of a row and a column | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_instances():
    mo.vstack([
        mo.md("**A class map and a mask for every object**"),
        mo.Html(C.instance_picture({
            "scene": "test scene 104", "classes": "its class map", "objects": "a mask per object",
            "kinds": ("circle", "square", "triangle"),
            "alt": "Test scene 104: two circles that overlap, and a small square. Its class map colors both circles as one area of the class circle. With a mask per object, each circle has its own mask with its own edge.",
        })),
        mo.md(r"""
        In the class map, the two circles are one area: from it alone, you cannot tell whether it holds one shape or two. With a mask per object, each circle is its own, and where they meet, every pixel belongs to one of them.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_stages():
    mo.vstack([
        mo.md("**The course's trained Mask R-CNN on the scene of the worked example**"),
        mo.Html(C.stages_picture({
            "boxes": "{n} boxes", "masks": "{n} masks",
            "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
            "alt": "The scene of the worked example twice. Left, its four true boxes, solid, and the 4 dashed boxes of 0.5 or more of the course's Mask R-CNN, each with its kind and score. Right, the scene in grey with the 4 masks it draws in those boxes, filled, and the edges of the true masks.",
        })),
        mo.md(r"""
        On the left, the boxes of the two-stage detector, now with RoIAlign; on the right, the mask the mask head draws in each. The large triangle's mask has an IoU of 0.92 with its true mask, the circle's 0.87 and the small triangle's 0.83. The square lies partly under the circle, so its true mask is a square with a round bite out of it; its mask has an IoU of 0.70.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_roi():
    mo.vstack([
        mo.md("**RoI pooling and RoIAlign of the circle's box**"),
        mo.Html(C.roi_picture({
            "pool": "RoI pooling: whole squares", "align": "RoIAlign: 64 points",
            "alt": "The circle of the scene of the worked example twice, with the squares of 8 by 8 pixels of the feature maps as thin lines and the circle's box dashed. Left, the 4 by 4 squares its corners lie in, framed: RoI pooling to 8 by 8 bins gives every square 2 by 2 bins. Right, the box cut into 8 by 8 bins, with a point at the middle of each.",
        })),
        mo.md(r"""
        RoI pooling of the circle's box to 8 by 8 bins takes the squares its corners lie in: 7 to 10 down and 6 to 9 across, 4 by 4 squares, larger than the box. Every bin is a whole square, and every square fills 2 by 2 bins: of the 64 numbers per channel, only 16 differ. RoIAlign cuts the box itself, and reads the feature maps at 64 points, each between four squares.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How well does a mask fit? The slider sets the IoU that a mask must be above to find its shape, from 0.5 to 0.95, as the ten thresholds of COCO. The course trained its Mask R-CNN twice with seed 0: once with RoI pooling, once with RoIAlign. On the left, the masks of both on the scene of the worked example: filled when they fit, dashed when they do not. On the right, the mask AP of both on the 200 test scenes at every threshold.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        t = float(t)
        _html, _stop = C.threshold_frame(t, {"pool": "RoI pooling: {n} above {t}", "align": "RoIAlign: {n} above {t}", "point": ".",
                                             "chart": "mask AP on the 200 test scenes", "x": "IoU threshold", "y": "mask AP",
                                             "align_line": "solid: RoIAlign", "pool_line": "dashed: RoI pooling",
                                             "chart_alt": "The mask AP of the course's Mask R-CNN on the 200 test scenes at every IoU threshold from 0.5 to 0.95: with RoIAlign above 0.8 up to a threshold of 0.8 and falling after it; with RoI pooling lower at every threshold, and much lower from 0.7 to 0.9.",
                                             "alt": "The scene of the worked example twice, with the masks of the course's Mask R-CNN trained with RoI pooling and with RoIAlign: filled when their IoU with their shape is above {t}, dashed when it is not. Next to them, the mask AP of both at every threshold."})
        return _html, f"At an IoU of {L.number(t, 2)}: mask AP {L.number(_stop['pool']['ap'], 3)} with RoI pooling, {L.number(_stop['align']['ap'], 3)} with RoIAlign; on this scene {_stop['pool']['fit']} and {_stop['align']['fit']} masks fit."

    explore_threshold = L.explore("threshold-slider", "IoU threshold", _frame, start=0.75, values=list(C.THRESHOLDS), role="hyper",
                                  task="Up to which threshold do both find almost every shape? From which threshold on does RoI pooling fall behind, and how far?")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 0.5, both find almost every shape: a mask AP of 0.924 with RoI pooling and 0.941 with RoIAlign. Then RoI pooling falls behind: at 0.75, 0.535 against 0.851; at 0.85, 0.238 against 0.740; at 0.9, 0.083 against 0.521. At 0.95 hardly a mask fits: 0.001 and 0.118. The mean of the ten, as COCO takes its mask AP, is 0.537 with RoI pooling and 0.758 with RoIAlign. The masks of RoI pooling are in the right place, but their edges are off: the median IoU of the masks of the shapes it finds is 0.88 for the circles, 0.80 for the squares and 0.73 for the triangles, against 0.95, 0.91 and 0.90 with RoIAlign; on the scene of the worked example, none of them is above 0.85. RoIAlign helps the boxes too: their mAP at an IoU of 0.75 is 0.879 with RoIAlign and 0.731 with RoI pooling. And RoI pooling's model adds a false box of the kind square, over the square and the circle, whose mask lies mostly on the circle.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        Mask R-CNN, on your two-stage detector:

        1. The backbone, the RPN and the proposals, as before. RoIAlign cuts every proposal to 4 by 4 bins for the second stage.
        2. For every proposal that learns a kind: RoIAlign to 8 by 8 bins, and the mask head: two convolutions of 3 by 3, a transposed convolution to 16 by 16, a convolution of 1 by 1 to a mask of logits for every kind.
        3. The target of the proposal: the true mask of its shape at the middle of each of its 16 by 16 bins.
        4. The loss of the mask of its kind: the sigmoid of every logit and the log loss, the mean over the 256 pixels. Train on the sum of the losses of the RPN, the second stage and the mask head.
        5. Detect: the boxes of the second stage after NMS; for every box, RoIAlign, the mask head, the mask of its kind, and `paste` it into the scene.
        """),
        L.watch_out("In training, the mask head draws in the proposals; to detect, in the boxes of the second stage, which fit better. Its mask is always pasted into the box it was drawn in."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example does steps 2 to 4 for the box of the circle, `s.shapes[3]`, on the scene of the worked example of the last lessons, `scene(18)`, with a network that has not trained yet: the backbone of your two-stage detector and the mask head. RoIAlign by hand: for each of 8 by 8 bins, the middle in pixels and then in squares, `int` the square above and to the left of it, `t` and `u` how far into that square it lies, and the mix of the four squares around it, as your `value_at`. The target: `int(top + (i + 0.5) * h / 16)` is the pixel that the middle of bin i lies in, and `mask[rows][:, cols]` takes those rows and columns of the true mask. The loss: the sigmoid and the log loss, written out. Run the cell as it is.
    """)
    return


@app.cell
def example_mask_head():
    import torch
    from torch import nn
    from everything_ai.scenes import scene

    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255

    # the backbone of your two-stage detector, and the mask head: two convolutions of 3 by 3,
    # a transposed convolution to twice the size, and a convolution of 1 by 1 to a mask for every kind
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
        maps = backbone(x)[0]   # the 64 feature maps of the scene
        top, left, bottom, right = s.shapes[3].box   # the circle
        h, w = bottom - top + 1, right - left + 1
        print(maps.shape, (top, left, bottom, right), h, w)

        # 2. RoIAlign to 8 by 8 bins: the middle of every bin, in squares, and the mix of the
        #    four squares around it
        aligned = torch.zeros(64, 8, 8)
        for i in range(8):
            for j in range(8):
                row = (top - 0.5 + (i + 0.5) * h / 8 - 3.5) / 8
                col = (left - 0.5 + (j + 0.5) * w / 8 - 3.5) / 8
                r, c = int(row), int(col)
                t, u = row - r, col - c
                above = (1 - u) * maps[:, r, c] + u * maps[:, r, c + 1]
                below = (1 - u) * maps[:, r + 1, c] + u * maps[:, r + 1, c + 1]
                aligned[:, i, j] = (1 - t) * above + t * below
        print(aligned.shape, row, col)
        print(aligned[0, 0])

        # 3. the target: the true mask at the pixel of the middle of each of 16 by 16 bins
        rows = [int(top + (i + 0.5) * h / 16) for i in range(16)]
        cols = [int(left + (j + 0.5) * w / 16) for j in range(16)]
        target = torch.tensor(s.shapes[3].mask[rows][:, cols]).float()
        print(target)
        print(target.sum())

        # 4. the mask head: a mask of 16 by 16 logits for every kind; the circle is kind 0,
        #    so only the first mask learns, with the sigmoid and the log loss
        logits = mask_head(aligned.unsqueeze(0))[0]
        p = 1 / (1 + torch.exp(-logits[0]))
        loss = -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()
        print(logits.shape, p.min(), p.max(), loss)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The feature maps have shape (64, 16, 16), and the circle's box, (58, 50, 86, 78), is 29 by 29 pixels. After the loops, `row` and `col` hold the middle of the last bin: 10.1484 and 9.1484 squares; the first lay at 6.9766 and 5.9766. RoI pooling would take the squares 7 to 10 down and 6 to 9 across; RoIAlign reads between the squares 6 and 11 down and 5 and 10 across. The first row of channel 0, (0.6619, 0.8114, 0.9698, 1.2657, 1.6095, 1.4621, 0.8869, 0.4258), has a different number in every bin.

    The target is a disc of ones that reaches the four edges, with zeros in the corners: 205 of its 256 pixels are circle. The mask head gives 3 masks of 16 by 16 logits. Untrained, the probabilities of the circle's mask all lie close to 0.5, from 0.4536 to 0.4695, and the loss is 0.7415: about the loss of a guess of 0.5 everywhere, log 2 = 0.693.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bilinear():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Bilinear interpolation of a whole grid

        The loops of the worked example mix four squares for one point at a time; `bilinear` does it for all points of a grid at once. Complete `bilinear(maps, ys, xs)`: `maps` is a tensor of shape (C, H, W), such as feature maps; `ys` and `xs` are lists of places, floats, counted in squares from the middle of square 0, as your `value_at` counted pixels. It gives back a tensor of shape (C, len(ys), len(xs)): the value of every channel at every point (y, x) of the grid, mixed from the four squares around it, the nearer counting more. A point outside the maps takes the value at the nearest edge: the given lines first move every place into the maps. `maps[:, r0]` takes the rows `r0` of every channel, and `[:, :, c0]` then the columns `c0` of those rows: the square above and to the left of every point at once. Make `t` a column, of shape (len(ys), 1), with `.reshape(-1, 1)`, and `u` a row: then broadcasting mixes every row with `u` and every column with `t`. The check tries the 2 by 2 pixels of your `value_at`, points outside them, maps of 2 channels and of 64, and whether training can follow your result back to the maps. When it is right, it saves `bilinear` in your project, as `work/projects/cnn/bilinear.py`; your `roi_align` and your `paste` use it.
        """),
        L.task("Make `bilinear` give the value of the maps at every point of the grid."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bilinear():
    box_bilinear = L.code_box("bilinear", lines=16)
    box_bilinear
    return (box_bilinear,)


@app.cell(hide_code=True)
def check_bilinear(box_bilinear):
    result_bilinear = L.check("bilinear", box_bilinear.value)
    result_bilinear
    return (result_bilinear,)


@app.cell(hide_code=True)
def hints_bilinear():
    L.hints(
        "bilinear",
        [
            "Your `value_at` of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en), for every point at once: for every y the row above it and the row under it, for every x the column to its left and to its right, how far the point lies from the first of each, and the mix of the four squares.",
            "`r0 = [int(y) for y in ys]` and `r1 = [min(r + 1, h - 1) for r in r0]`, the same for the columns. `t = (torch.tensor(ys) - torch.tensor(r0)).reshape(-1, 1)`, and `u` the same for `xs` and `c0`, without `reshape`.",
            "`top = (1 - u) * maps[:, r0][:, :, c0] + u * maps[:, r0][:, :, ___]`, `bottom` the same with `r1`, and `return (1 - t) * top + t * ___`.",
        ],
        why="For the point (0.25, 0.5) of the maps [[0, 100], [200, 40]]: halfway along row 0 is 50, halfway along row 1 is 120, and a quarter of the way from 50 to 120 is 50 + 0.25 × 70 = 67.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_align():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. RoIAlign

        Complete `roi_align(maps, box, k)`, step 2 of the recipe for any box and any k: `maps` holds the feature maps of one scene, of shape (C, H, W), every square 8 by 8 pixels; `box` is (top, left, bottom, right) in pixels, both ends included, floats, and may stick out of the scene. It gives back a tensor of shape (C, k, k): the value of the feature maps at the middle of each of k by k bins of the box, with your `bilinear`, which the first line imports from your project. The box covers `bottom - top + 1` pixels down, from `top - 0.5`. The check tries five boxes on made-up maps, two of them partly or wholly outside the maps and one inside a single square, and the circle of the scene of the worked example on maps of the mean color of every square. When it is right, it saves `roi_align` in your project, as `work/projects/cnn/roi_align.py`.
        """),
        L.task("Make `roi_align` give every box k by k numbers per channel, without rounding."),
    ])
    return


@app.cell(hide_code=True)
def exercise_align():
    box_roi_align = L.code_box("roi-align", lines=10)
    box_roi_align
    return (box_roi_align,)


@app.cell(hide_code=True)
def check_align(box_roi_align):
    result_align = L.check("roi-align", box_roi_align.value)
    result_align
    return (result_align,)


@app.cell(hide_code=True)
def hints_align():
    L.hints(
        "roi-align",
        [
            "The loops of the worked example, as lists: the middles of the k bins down and across the box, in pixels, then in squares, and your `bilinear` at all of them at once.",
            "`h, w = bottom - top + 1, right - left + 1`, then `ys = [top - 0.5 + (i + 0.5) * h / k for i in range(k)]`, the middles in pixels, and the same for `xs`.",
            "`return bilinear(maps, [(y - ___) / 8 for y in ys], [(x - ___) / 8 for x in xs])`.",
        ],
        why="For the first box, (5.5, 9.0, 30.2, 44.0), with 2 by 2 bins: it is 25.7 pixels high, from 5.0, so the middles of its bins lie at the rows 11.425 and 24.275, or 0.9906 and 2.5969 squares. RoI pooling took the whole squares 0 and 1 for the first bin and 2 and 3 for the second.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_paste():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Paste a mask into the scene

        Complete `paste(mask, box, size)`, the end of step 5: `mask` holds the probabilities the mask head gives one box, of shape (m, m); `box` is (top, left, bottom, right) in pixels, both ends included, and may stick out of the scene or begin between two pixels; the scene is `size` by `size` pixels. It gives back `out`, a tensor of shape (size, size), True where the mask, made as large as the box with your `bilinear`, is 0.5 or more. Take every pixel whose middle lies in the box, from `top - 0.5` to `bottom + 0.5`, and in the scene; its middle lies `r - top + 0.5` pixels into the box, which is `bottom - top + 1` pixels high, and so at that part of the mask's m rows, counted from the top edge of the mask; `bilinear` counts from the middle of its first row. The check tries a mask of 4 by 4 in boxes of a scene of 8 by 8, one of which sticks out and one of which does not begin at a whole pixel, and two masks of 16 by 16 in a scene of 128 by 128. When it is right, it saves `paste` in your project, as `work/projects/cnn/paste.py`.
        """),
        L.task("Make `paste` put the mask of a box into the scene."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paste():
    box_paste = L.code_box("paste", lines=17)
    box_paste
    return (box_paste,)


@app.cell(hide_code=True)
def check_paste(box_paste):
    result_paste = L.check("paste", box_paste.value)
    result_paste
    return (result_paste,)


@app.cell(hide_code=True)
def hints_paste():
    L.hints(
        "paste",
        [
            "RoIAlign the other way round: for every pixel of the scene whose middle lies in the box, where that middle lies in the mask, and the value of the mask there with your `bilinear`; 0.5 or more belongs to the object.",
            "The first row is `max(math.ceil(top - 0.5), 0)` and the last `min(math.floor(bottom + 0.5), size - 1)`, the same for the columns. Row r lies `(r - top + 0.5) / (bottom - top + 1) * m` rows into the mask, and `- 0.5` counts that from the middle of its first row.",
            "`ys = [(r - top + 0.5) / (bottom - top + 1) * m - 0.5 for r in range(r0, r1 + 1)]`, the same for `xs`, then `out[r0:r1 + 1, c0:c1 + 1] = bilinear(mask.unsqueeze(0), ys, xs)[0] >= ___`.",
        ],
        why="In the first case, the box covers the rows 1 to 6, 6 pixels for the 4 rows of the mask. The middle of pixel row 1 lies 0.5 pixels into the box, at 0.5 / 6 × 4 − 0.5 = −0.17 rows of the mask: above the middle of its first row, so it takes the value of row 0. The box is 4 pixels wide, so every pixel column lands on the middle of a column of the mask.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_learns():
    mo.md(r"""
    ### Exercise 4. What the mask head learns from a proposal
    """)
    return


@app.cell(hide_code=True)
def choice_learns_cell():
    choice_learns = L.choice(
        "learns",
        question="In training, a proposal overlaps the circle with an IoU of 0.62, and no other shape by 0.5 or more. The mask head draws 3 masks for it, one for every kind. What does it learn from this proposal?",
        options={
            "right": "Only the circle's mask learns: the true mask of the circle, at the 16 by 16 bins of the proposal",
            "all": "All three: the circle's mask the circle, and the masks of the square and the triangle 0 everywhere",
            "box": "Only the circle's mask learns: the true mask of the circle, at 16 by 16 bins of the circle's own box",
            "none": "Nothing: only the true boxes teach the mask head",
        },
        correct="right",
        why={
            "right": "Its kind is circle, so only the circle's mask counts in the loss; the other two masks do not learn from it. Its target is the circle's mask at the middles of the bins of the proposal itself, the box the mask head looks at: a proposal that is a little too large learns a circle that does not reach its edges.",
            "all": "Then the masks of the kinds compete for every pixel, as with softmax over the kinds in a class map. Mask R-CNN lets only the mask of the box's kind learn; with softmax, the paper's masks were worse, a mask AP of 24.8 instead of 30.3.",
            "box": "The mask head sees the features of the proposal, so its target is cut to the proposal: the pixel at the middle of each of its 16 by 16 bins. In the circle's own box, the mask would lie where the head did not look.",
            "none": "Every proposal with an IoU of 0.5 or more with a true box learns a mask, as it learns a kind; the true boxes only join the proposals, as for the second stage.",
        },
    )
    choice_learns
    return (choice_learns,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains the course's Mask R-CNN, the two-stage detector with a mask head, with your `roi_align`, and through it your `bilinear`, and with your `proposals`, for 8 epochs on the 2,000 training scenes, in mini-batches of 32. Then it finds the shapes of the 200 test scenes with your `boxes_from`, and pastes their masks with your `paste`. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; that takes a while: on the course's computer, about eleven minutes. While it trains, the chart shows the training loss of the three parts together every 10 batches. Under the chart are the masks it gives three test scenes at that moment, and under them the feature maps of the last layer of its backbone, which all three parts read, on the scene of the worked example; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train Mask R-CNN")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_mask_rcnn(train_button, result_bilinear, result_align, result_paste):
    result_bilinear, result_align, result_paste  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train Mask R-CNN.*")
    else:
        _now = {}   # the live chart, made at the first report
        _words = {"scene": ("scene {seed}: {n} mask", "scene {seed}: {n} masks"), "layer": "Layer {name} of Mask R-CNN on scene {seed}: the maps all three parts read",
                  "alt": "Three test scenes in grey, each with the edges of its true masks and the masks of the boxes of 0.5 or more that Mask R-CNN gave it after that epoch, filled."}
        _words_now = _words | {"alt": "Three test scenes in grey, each with the edges of its true masks and the masks of the boxes of 0.5 or more that Mask R-CNN gives it now, filled."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Mask R-CNN: the training loss", "x": "epoch", "y": "training loss",
                                        "alt": "The training loss of Mask R-CNN so far, its three parts together, every 10 batches."}, _p["epochs"], low=0.0)
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-mask-now"))
            if "mask" in _p:   # after every epoch: its mean
                _loss = _p["rpn"] + _p["second"] + _p["mask"]
                _now["chart"].line(f"**Mask R-CNN** · epoch {_p['epoch']} of {_p['epochs']} · training loss {L.number(_loss, 3)}")
            else:   # every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Mask R-CNN** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · training loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `roi_align` and your `paste` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than twenty-five minutes. Does your `bilinear`, your `roi_align` or your `proposals` do more work than it needs, such as a loop over the points of the grid? `bilinear` can mix all points of the grid at once.*",
                "roi_align": "*Your `roi_align` in `work/projects/cnn/roi_align.py`, or the `bilinear` it uses, stops with an error, or does not give 64 × k × k numbers that training can follow back. Did you change one of their files? Delete it, and check its exercise again: its check then saves your code again.*",
                "paste": "*Your `paste` in `work/projects/cnn/paste.py` stops with an error, or does not give a mask of 128 by 128 pixels. Did you change its file? Delete it, and check exercise 3 again.*",
                "proposals": "*Your `proposals` in `work/projects/cnn/proposals.py` stops with an error, or does not give between 1 and 32 boxes for a scene. Did you change its file? Delete it, and check its exercise in [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en) again.*",
                "offsets": "*Your `boxes_from` in `work/projects/cnn/offsets.py` stops with an error on the detector's offsets. Did you change its file? Delete it, and check its exercise in [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss of each part", "x": "epoch", "y": "loss", "rpn": "RPN, solid", "second": "second stage, dashed",
                       "mask": "mask head, with rings",
                       "alt": "The training loss of each part after every epoch: of the RPN solid, of the second stage dashed, of the mask head solid with rings."}
            _curves, _moments = _result["curves"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_m) for _m in _moments[_e - 1])
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"After epoch {_e}: loss {L.number(_curves['rpn'][_e - 1], 3)} for the RPN, {L.number(_curves['second'][_e - 1], 3)} for the second stage and {L.number(_curves['mask'][_e - 1], 3)} for the mask head, and {_n} {'mask' if _n == 1 else 'masks'} on these three scenes, which have {C.SHOWN_SHAPES} shapes.")

            if len(_moments) == len(_curves["mask"]) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_moments), values=list(range(1, len(_moments) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: after which epoch do the masks take the shape of their object?")]
            else:
                _pic = [mo.Html(C.curves_picture(_curves, _labels, L.lang))] if _curves["mask"] else []
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("circles", "squares", "triangles"), _result["aps"]))
            _medians = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("circles", "squares", "triangles"), _result["median"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Your Mask R-CNN, with {C.whole(_result['params'], L.lang)} parameters, finds {_result['found']} of the {_result['shapes']} shapes of the 200 test scenes with a box of their kind, with a score of 0.5 or more and an IoU above 0.5; {_result['false']} of its boxes of 0.5 or more find no shape. The median IoU of the masks of the shapes it finds: {_medians}. Its mask AP at an IoU of 0.5 is {L.number(_result['map'], 3)}, with an AP of {_aps}; at 0.75, it is {L.number(_result['map75'], 3)}. The course's Mask R-CNN, trained the same way with the seeds 0 to 4, has a mask AP of 0.941 to 0.948 at an IoU of 0.5, and 0.846 to 0.857 at 0.75; with RoI pooling instead of RoIAlign, 0.915 to 0.925, and 0.535 to 0.592 at 0.75.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 5. Why rounding hurts a mask
    """)
    return


@app.cell(hide_code=True)
def explain_why_cell():
    explain_why = L.explain(
        "why-align",
        question="Trained with RoI pooling, the course's Mask R-CNN finds almost as many shapes at an IoU of 0.5 as with RoIAlign, but at 0.75 its mask AP falls from about 0.85 to below 0.6. Why does rounding to whole squares hurt the masks so much more than the kinds? And what does RoIAlign do instead?",
        model_answer="RoI pooling takes the whole squares that the corners of a box lie in, so its numbers do not begin where the box begins: they can lie almost a whole square, 8 pixels, off, and boxes a few pixels apart get the same numbers. The mask is pasted into the exact box, so the mask head must guess where in its bins the edges of the shape lie, and a mask whose edge is a few pixels off loses much of its IoU. The kind of a box only needs what is in it, and a few pixels do not change that. RoIAlign takes the middle of every bin of the box itself, in squares without rounding, and the value there with bilinear interpolation of the four squares around it: its numbers move with the box, and every bin gets its own numbers. Of 8 by 8 bins on the circle's box, RoI pooling gave only 16 different numbers per channel.",
        points=[
            "you say that RoI pooling rounds a box to the whole squares its corners lie in, so its numbers can lie up to a square off the box, and boxes a few pixels apart get the same numbers",
            "you say that a mask must place its edges to the pixel, while the kind of a box hardly changes when you look a few pixels off",
            "you say that RoIAlign reads the feature maps at the exact middles of the bins of the box, mixed from the four squares around each, so that its numbers move with the box",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The two-stage detector of [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en) has a third head now: a small FCN, as in [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en), that draws a mask in every box, so that every object gets a mask of its own, also when it touches another of its kind. Your `bilinear` reads feature maps at any point, as your `value_at` of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en) read an image; your `roi_align` uses it to cut every box to k by k bins without rounding; and your `paste` puts the small mask of a box back into the scene. On the course's scenes, RoIAlign made the masks fit much better than RoI pooling, and the boxes too: at an IoU of 0.75, its mask AP rose from between 0.535 and 0.592 to between 0.846 and 0.857, with the seeds 0 to 4. Your project `cnn` now has `bilinear.py`, `roi_align.py` and `paste.py`. In the next lesson, *Segment what you point at: SAM 2.1*, a large pretrained model draws the mask of whatever you point at, also of things it has no name for.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_bilinear, result_align, result_paste, choice_learns, explain_why):
    L.wrap_up(result_bilinear, result_align, result_paste, choice_learns, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
