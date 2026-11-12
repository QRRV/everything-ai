# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-rcnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-rcnn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="The head of your one-stage detector gives each of the 3 anchors of a square 4 logits and 4 offsets: 24 channels. A head that only tells *object* from *background*, for 9 anchors on every square, is a convolution of 3 by 3 too. How many output channels does it need?",
        options={
            "right": "54: 9 anchors times 2 logits and 4 offsets",
            "logits": "18: 9 anchors times 2 logits",
            "one": "6: 2 logits and 4 offsets, for the square",
            "kinds": "72: 9 anchors times 4 logits and 4 offsets",
        },
        correct="right",
        why={
            "right": "Every anchor gets its own 2 logits and its own 4 offsets: 9 × 6 = 54. That is the head of Faster R-CNN's region proposal network: 2 × 9 scores and 4 × 9 numbers for the boxes. Today's RPN gives 3 × 6 = 18.",
            "logits": "Without offsets, every box stays its anchor. Each anchor also needs its 4 offsets: 9 × (2 + 4). See [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).",
            "one": "That is one anchor per square. Each of the 9 anchors gets its own logits and its own offsets: 9 × 6.",
            "kinds": "4 logits are for background and 3 kinds. To tell *object* from *background*, 2 logits are enough: 9 × (2 + 4). See [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def quick_check_many():
    quick_many = L.choice(
        "quick-many",
        question="Before NMS, your one-stage detector gave the scene of its worked example 17 boxes with a score of 0.5 or more, for 4 shapes. Why so many?",
        options={
            "right": "Several anchors around each shape learned that shape, each with its own offsets",
            "low": "A score of 0.5 is too low: at 0.9 every shape would get one box",
            "backbone": "The backbone sees every shape at three sizes, one for each anchor size",
        },
        correct="right",
        why={
            "right": "Every anchor with an IoU of 0.5 or more with a shape learned it: 8 anchors learned the circle. Trained, the detector gave the circle 7 boxes of 0.5 or more, and NMS kept the one with the highest score. Today, NMS helps twice: once for the proposals, once for the boxes at the end.",
            "low": "Before NMS, the circle had 7 boxes, with scores from 0.70 to 0.99, and 5 of them above 0.94: a higher score would still leave several. See [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).",
            "backbone": "The backbone makes one set of feature maps, the same for every anchor. The boxes come from the anchors that learned each shape. See [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en).",
        },
    )
    quick_many
    return (quick_many,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your one-stage detector decides about every anchor in one pass, from the squares around it. A **two-stage detector** looks twice. First it proposes regions that may hold an object; then it looks at each proposal again, at the features of the whole region, and gives it a kind and a better box. That is the family of R-CNN, which led object detection for years. R-CNN (Girshick and colleagues, 2014) took about 2,000 proposals of every photo from *selective search*, a method without learning that joins pixels of similar color and texture into regions; it cut each of them from the photo, stretched it to 227 by 227 pixels, and ran a whole CNN on it. Fast R-CNN (Girshick, 2015) ran the CNN once over the whole photo, and cut every proposal from its feature maps. Faster R-CNN (Ren and colleagues, 2015) also made the proposals with the network itself: a second, small head on the same feature maps. Today you build that last one on the course's scenes, with the backbone of your one-stage detector, and compare the two.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_seconds():
    guess_seconds = L.guess(
        "seconds",
        question="R-CNN ran a whole CNN on each of about 2,000 regions of a photo. How long did that take per photo, on a GPU of 2014?",
        options={
            "fast": "About 0.05 seconds",
            "half": "About half a second",
            "right": "About 13 seconds",
            "minutes": "About two minutes",
        },
        reveal="About 13 seconds: the paper of R-CNN gives 13 seconds per photo on a GPU, and 53 on a CPU, for the proposals and their features, with the CNN it used. With the deeper VGG16, R-CNN took 47 seconds, as the paper of Fast R-CNN measured; Fast R-CNN took 0.32 seconds with it, not counting the proposals, while selective search alone took about 2 seconds on a CPU. Faster R-CNN, with VGG16 too, took 0.2 seconds for everything: 5 photos per second on a GPU.",
    )
    guess_seconds
    return (guess_seconds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The first stage proposes.** It is your one-stage detector with two classes: *object* and *background*. It has the same backbone, and a head of 3 by 3 that gives each of the 3 anchors of a square 2 logits and 4 offsets: 18 channels. It learns as your detector did, with the targets of your `targets`, in which every kind is class 1, and the same two losses. Faster R-CNN's RPN draws its lines at 0.7 and 0.3 instead of 0.5 and 0.4, and learns from 256 of the anchors of a photo at a time; here it learns just as your detector did. This head is the **region proposal network**, RPN for short. Faster R-CNN has 9 anchors on every square, of 3 sizes and 3 shapes, so its RPN gives 2 × 9 scores and 4 × 9 numbers for the boxes.

        **From 768 anchors to 32 proposals.** Softmax over its 2 logits gives every anchor a probability of an object, and your `boxes_from` makes its box. The 100 anchors with the highest probability go into NMS at 0.7, and the first 32 boxes that stay are the **proposals** of the scene. The threshold is 0.7, as in Faster R-CNN, not 0.5: it leaves a few different proposals around every shape for the second stage to make its boxes from, and the NMS at the end removes the copies. Faster R-CNN keeps about 2,000 proposals per photo in training and 300 to detect.

        **RoI pooling: every proposal the same size.** A proposal can be large or small, and the second stage needs the same number of inputs for every one. **RoI pooling** (RoI: region of interest) takes the squares of the feature maps that the proposal's corners lie in, rows `int(top // 8)` to `int(bottom // 8)` and columns `int(left // 8)` to `int(right // 8)`, cuts those squares into 4 by 4 bins, as evenly as it can, and keeps the highest number of each bin, for each of the 64 channels. Every proposal, of any size, becomes 64 × 4 × 4 = 1,024 numbers. A bin of n rows starts at row ⌊i · n / 4⌋ and stops before row ⌈(i + 1) · n / 4⌉, for i from 0 to 3: with 6 rows, the bins take rows 0–1, 1–2, 3–4 and 4–5, and with 3 rows, rows 0, 0–1, 1–2 and 2. Where the rows do not divide evenly, neighboring bins share a row. `F.adaptive_max_pool2d(region, 4)` does exactly this. Fast R-CNN pools every proposal to 7 by 7 on the maps of VGG16.

        **The second stage looks again.** The 1,024 numbers of a proposal go through a linear layer of 128 numbers, ReLU, and a linear layer to 8 numbers: 4 logits, for background and the 3 kinds, and 4 offsets, now from the proposal to its box. A proposal learns as an anchor did, but with one threshold: the kind of the true box it overlaps most, from an IoU of 0.5 on, and the offsets to that box; below 0.5, background. Faster R-CNN gives every kind its own 4 offsets; here one set for all kinds is enough. In training, the true boxes join the proposals, as the code of Faster R-CNN does, so that the second stage also sees boxes that fit from the first epoch on.

        **One loss for both stages.** The detector trains on the loss of the RPN plus the loss of the second stage, each a cross-entropy plus a mean absolute difference of the offsets. The proposals count as given numbers: training follows the loss back through RoI pooling into the feature maps, but not into the corners of the proposals. The makers of Faster R-CNN call this *approximate joint training*; it took 25 to 50% less training time than their first way, which trained the two stages in turns.

        **Detecting.** A scene, its feature maps, the RPN, 32 proposals, RoI pooling, the second stage. The score of a proposal is its highest probability of a kind, its box comes from `boxes_from` with the proposal as the anchor, and NMS for each kind at 0.5 keeps the best boxes, as in your one-stage detector.
        """),
        L.watch_out("RoI pooling rounds. A proposal from row 41.4 to 67.7 takes the squares 5 to 8, rows 40 to 71 of the scene. Every proposal whose corners lie in the same squares gets the same 1,024 numbers, so the second stage cannot see how far into a square a proposal begins. Mask R-CNN, in a later lesson, replaces it with RoIAlign, which does not round."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **two-stage detector** | a detector that first proposes regions that may hold an object, and then looks at each proposal again to give it a kind and a better box. R-CNN, Fast R-CNN and Faster R-CNN are two-stage detectors | `TwoStage` |
        | **region proposal network** (RPN) | the first stage of Faster R-CNN: a head on the feature maps that gives every anchor a probability of an object and 4 offsets; its surest boxes, after NMS at 0.7, are the proposals | `self.rpn` |
        | **proposal** | a box that the first stage thinks may hold an object; the second stage looks at it again | `proposals` |
        | **RoI pooling** | cutting the features of a region of interest, such as a proposal, to the same size: its squares in k by k bins, and the highest number of each bin | `roi_pool` |
        | **Faster R-CNN** | the two-stage detector of Ren and colleagues (2015), whose proposals come from an RPN on the same feature maps as the second stage | |
        | `F.adaptive_max_pool2d(x, k)` | cuts every map of x into k by k bins as evenly as it can, and keeps the highest number of each bin: k by k numbers per map, whatever its size | |
        | `t.argsort(descending=True)` | the places of the numbers of a tensor, from the highest number to the lowest; without `descending=True`, from the lowest | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_stages():
    mo.vstack([
        mo.md("**The course's trained two-stage detector on the scene of the worked example**"),
        mo.Html(C.stages_picture({
            "first": "first stage: {n} proposals",
            "second": "second stage: {n} boxes",
            "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
            "alt": "The scene of the worked example twice, with its four true boxes, solid. Left, the 32 thin dashed proposals of the first stage, all on or around the four shapes, most of them around the square. Right, the 4 boxes of 0.5 or more that the second stage makes of them, after NMS, each with its kind and score.",
        })),
        mo.md(r"""
        On the left, the 32 proposals of the first stage. Every one of them lies on or around a shape: 18 around the square, which the circle partly covers, 6 around the small triangle, and 4 each around the large triangle and the circle. Only the first 8 have a probability of an object above 0.7, and the last ones 0.007, but they all go on: the second stage decides. On the right, what it makes of them: for every proposal a kind, a score and a better box, then NMS for each kind. All four shapes get a box, also the small triangle, which your one-stage detector gave no box on this scene.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_roi():
    mo.vstack([
        mo.Html(C.roi_picture({
            "title": "RoI pooling of the first proposal",
            "alt": "Part of the scene of the worked example, with the squares of 8 by 8 pixels of the feature maps as thin lines. The first proposal, on the large triangle, is dashed; the squares its corners lie in have a solid frame, cut into 4 by 4 bins.",
        })),
        mo.md(r"""
        The first proposal, the one the RPN is surest of, lies on the large triangle. Its corners lie in 4 by 4 squares, so here every bin is one square, and the 1,024 numbers are the 64 channels of those 16 squares. The solid frame is larger than the proposal: RoI pooling takes whole squares.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How many proposals does the second stage need? The slider sets how many proposals the course's trained detector keeps per scene to detect, from 1 to 64; it trained with 32. On the left, its boxes of 0.5 or more on the scene of the worked example; on the right, its mAP on the 200 test scenes at an IoU of 0.5, solid, and of 0.75, dashed.
    """)
    return


@app.cell(hide_code=True)
def explore_keep_cell():
    def _frame(n):
        n = int(n)
        _html, _stop = C.keep_frame(n, {"title": "{n} proposals: {boxes} boxes", "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
                                        "chart": "mAP on the 200 test scenes", "keep": "proposals kept per scene",
                                        "at50": "IoU 0.5", "at75": "IoU 0.75",
                                        "chart_alt": "The mAP of the course's two-stage detector on the 200 test scenes for each number of proposals kept per scene, at an IoU of 0.5 and of 0.75: low with 1 and 2 proposals; at 0.5 close to the highest from 8 on, at 0.75 the highest at 8 and lower after it.",
                                        "alt": "The scene of the worked example with its four true boxes and the {boxes} dashed boxes of 0.5 or more of the detector when it keeps {n} proposals. Next to it, its mAP for every number of proposals."})
        return _html, f"{n} proposals per scene: {_stop['covered']} of the {C.TRAINED['shapes']} shapes have a proposal with an IoU above 0.5; mAP {L.number(_stop['map'], 3)} at an IoU of 0.5 and {L.number(_stop['map75'], 3)} at 0.75."

    explore_keep = L.explore("keep-slider", "proposals kept per scene", _frame, start=32, values=list(C.KEEPS), role="hyper",
                             task="From how many proposals on does almost every shape have one? And what happens to the mAP at 0.75 after that?")
    explore_keep
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        With 1 proposal per scene, the detector can find at most one shape per scene: 203 of the 808 shapes have a proposal with an IoU above 0.5, a few more than the 200 scenes, because in a few scenes one proposal covers two shapes that lie on each other; the mAP is 0.241. With 4, 717 shapes have one; with 8, 787; with 32, 805. The mAP at 0.5 hardly changes after 16 proposals. At 0.75, it is highest at 8, 0.824, and falls to 0.751 at 64: more proposals give a shape more boxes, and proposals whose corners lie in the same squares get the same score and the same offsets, wherever in those squares they begin, so NMS may keep one that fits less well. The course trained the detector again with RoIAlign instead of RoI pooling, with seed 0: then the mAP at 0.75 stays between 0.875 and 0.892 from 8 proposals on.
        """),
        L.watch_out("Do not choose the number of proposals from this chart: it measures the test scenes, and a number chosen on them makes their mAP promise more than new scenes would give. Choose it on a validation set, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en). This lesson keeps 32 to detect, as many as in training."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        A two-stage detector, as Faster R-CNN:

        1. The backbone makes feature maps of the scene, and the RPN, a convolution of 3 by 3, gives every anchor 2 logits and 4 offsets.
        2. Proposals: the probability of an object of every anchor, its box from its offsets, the 100 surest anchors, NMS at 0.7, and the first 32 boxes that stay.
        3. RoI pooling: for every proposal, the squares its corners lie in, cut into 4 by 4 bins, and the highest number of each bin, for every channel.
        4. The second stage: the 1,024 numbers of a proposal in one row, a linear layer, ReLU, and a linear layer to 4 logits and 4 offsets from the proposal.
        5. Train both at once: the RPN learns as a one-stage detector with 2 classes; every proposal, and every true box, learns the kind + 1 of the true box it overlaps most from an IoU of 0.5 on, else background, and the offsets to that box. The loss is the sum of the two losses; the proposals count as given numbers.
        6. Detect: the score of every proposal, its box from `boxes_from` with the proposal as the anchor, and NMS for each kind at 0.5.
        """),
        L.watch_out("The offsets of the second stage start from the proposal, not from an anchor. A proposal that already fits needs offsets near 0; one that is too large needs negative ones for its height and width. `boxes_from(proposals, offsets)` turns them into boxes."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example does steps 1 to 4 on the scene of the worked example of the last two lessons, `scene(18)`, with a network that has not trained yet. The backbone is the one of your one-stage detector. In step 2, the probability of an object is softmax over the 2 logits of every anchor, written out as in the last lesson; the boxes are your `boxes_from` of [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en), written out too; and `nms` from `everything_ai.scenes` gives what your `nms` of [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en) gave, the places of the boxes it keeps, the highest score first, quickly for a few hundred boxes. `argsort(descending=True)` gives the places of the anchors from the highest probability to the lowest, and `[:100]` keeps the first 100. Step 3 pools the true box of the square, `s.shapes[2]`, by hand: `math.floor` rounds down and `math.ceil` rounds up, and `.max(dim=1)` gives the highest number of every row of `region[:, a:b, c:d].reshape(64, -1)`, the squares of one bin for each channel, and where it lies. Then `F.adaptive_max_pool2d` does the same in one step. Run the cell as it is.
    """)
    return


@app.cell
def example_two_stages():
    import math
    import torch
    import torch.nn.functional as F
    from torch import nn
    from everything_ai.scenes import anchor_boxes, nms, scene

    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    anchors = torch.tensor(anchor_boxes()).float()

    # 1. the backbone of your one-stage detector, and the RPN: a convolution of 3 by 3 that gives
    #    each of the 3 anchors of a square 2 logits (background, object) and 4 offsets
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    )
    rpn = nn.Conv2d(64, 3 * 6, 3, padding=1)
    second = nn.Sequential(nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 8))
    with torch.no_grad():
        maps = backbone(x)[0]   # the 64 feature maps of the scene
        rows = rpn(maps.unsqueeze(0)).permute(0, 2, 3, 1).reshape(768, 6)
        print(maps.shape, rows.shape)

        # 2. the proposals: the probability of an object, the boxes, the 100 surest anchors,
        #    NMS at 0.7, and the first 32 boxes that stay
        e = torch.exp(rows[:, :2])
        score = e[:, 1] / e.sum(dim=1)
        middle_a, size_a = (anchors[:, :2] + anchors[:, 2:]) / 2, anchors[:, 2:] - anchors[:, :2] + 1
        middle, size = middle_a + rows[:, 2:4] * size_a, size_a * torch.exp(rows[:, 4:])
        boxes = torch.cat([middle - (size - 1) / 2, middle + (size - 1) / 2], dim=1)   # your boxes_from
        best = score.argsort(descending=True)[:100]
        kept = nms(boxes[best], score[best], 0.7)
        props = boxes[best][kept[:32]]
        print(score.max(), score.min(), len(kept), props.shape)
        print(props[0])

        # 3. RoI pooling of one box, the square's: the squares its corners lie in, cut into
        #    4 by 4 bins, and the highest number of each bin, for every channel
        top, left, bottom, right = s.shapes[2].box
        r0, c0, r1, c1 = int(top // 8), int(left // 8), int(bottom // 8), int(right // 8)
        region = maps[:, r0:r1 + 1, c0:c1 + 1]
        h, w = region.shape[1], region.shape[2]
        pooled = torch.zeros(64, 4, 4)
        for i in range(4):
            for j in range(4):
                a, b = math.floor(i * h / 4), math.ceil((i + 1) * h / 4)
                c, d = math.floor(j * w / 4), math.ceil((j + 1) * w / 4)
                highest, _ = region[:, a:b, c:d].reshape(64, -1).max(dim=1)
                pooled[:, i, j] = highest
        print((top, left, bottom, right), (r0, c0, r1, c1), region.shape)
        print(pooled[0])
        print((pooled - F.adaptive_max_pool2d(region, 4)).abs().max())

        # 4. the second stage: the 64 x 4 x 4 numbers in one row, then 8 numbers: 4 logits
        #    (background, circle, square, triangle) and 4 offsets from the box
        out = second(pooled.reshape(1, -1))
        print(out.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The feature maps have shape (64, 16, 16), and the RPN gives 768 rows of 6. Untrained, the probabilities of an object run from 0.0932 to 0.8329. NMS at 0.7 keeps 95 of the 100 surest boxes: two anchors never overlap with an IoU above 0.6, so it is the random offsets of the untrained RPN that made 5 boxes overlap a surer one by more than 0.7. The first proposal is about (60.0, 43.4, 88.8, 84.2).

    The box of the square, (42, 39, 73, 72), lies in the squares 5 to 9 down and 4 to 9 across: a region of 5 by 6 squares, shape (64, 5, 6). Its 4 bins of rows take rows 0–1, 1–2, 2–3 and 3–4 of the region, and its 4 bins of columns take columns 0–1, 1–2, 3–4 and 4–5. `pooled[0]` holds the 16 highest numbers of channel 0, the first row (1.2122, 1.2122, 1.1315, 1.1315): the first two bins share column 1 of the region, and the highest number of both lies there; the last two share column 4. And `F.adaptive_max_pool2d` gives exactly the same: the largest difference is 0. The second stage gives 1 row of 8 numbers for the box.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fast():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What made Fast R-CNN fast
    """)
    return


@app.cell(hide_code=True)
def choice_fast_cell():
    choice_fast = L.choice(
        "fast",
        question="Fast R-CNN found the objects of a photo 146 times faster than R-CNN, not counting the proposals, with the same deep CNN, VGG16, and the same proposals of selective search. What made the difference?",
        options={
            "right": "It ran the CNN once over the whole photo, and cut every proposal from its feature maps",
            "fewer": "It looked at far fewer proposals per photo",
            "small": "It shrank every proposal to 7 by 7 pixels before the CNN",
            "rpn": "It made the proposals with a region proposal network",
        },
        correct="right",
        why={
            "right": "R-CNN ran the CNN about 2,000 times per photo, once for every proposal. Fast R-CNN runs it once, and RoI pooling cuts each proposal from the feature maps to 7 by 7 bins. Your detector does the same with 4 by 4.",
            "fewer": "Fast R-CNN still took about 2,000 proposals per photo from selective search. What changed is how often the CNN runs.",
            "small": "The 7 by 7 is the size RoI pooling gives every proposal on the feature maps, after the CNN, not pixels before it. The CNN ran once, over the whole photo.",
            "rpn": "That came later, with Faster R-CNN. Fast R-CNN still took its proposals from selective search, about 2 seconds per photo on a CPU.",
        },
    )
    choice_fast
    return (choice_fast,)


@app.cell(hide_code=True)
def text_exercise_roi():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. RoI pooling

        Complete `roi_pool(maps, box, k)`, step 3 of the recipe: `maps` holds the feature maps of one scene, a tensor of shape (C, H, W), every square 8 by 8 pixels; `box` is (top, left, bottom, right) in pixels, floats, and may stick out of the scene. It gives back a tensor of shape (C, k, k): the highest number of each of k by k bins of the squares that the corners of the box lie in. The square of a pixel is `int(pixel // 8)`; keep every square from 0 to the last square of the maps, `H - 1` down and `W - 1` across, so that a box that sticks out still has squares. Cut the bins with `F.adaptive_max_pool2d`, or by hand, as in the worked example: then the demo trains a few times slower. The check tries five boxes: three on made-up maps of 6 by 6 squares, one of which sticks out above and to the right and one of which lies inside a single square, one on maps of 16 by 16, and one below maps of 12 by 10. When it is right, it saves `roi_pool` in your project, as `work/projects/cnn/roi_pool.py`.
        """),
        L.task("Make `roi_pool` give every box k by k numbers per channel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_roi():
    box_roi_pool = L.code_box("roi-pool", lines=10)
    box_roi_pool
    return (box_roi_pool,)


@app.cell(hide_code=True)
def check_roi(box_roi_pool):
    result_roi = L.check("roi-pool", box_roi_pool.value)
    result_roi
    return (result_roi,)


@app.cell(hide_code=True)
def hints_roi():
    L.hints(
        "roi-pool",
        [
            "Step 3 of the worked example, as a function: the squares of the corners, the region of the maps between them, both ends included, and its k by k bins.",
            "`h, w = maps.shape[1], maps.shape[2]`, then each square as `min(max(int(top // 8), 0), h - 1)`: `max` lifts a square below 0 to 0, and `min` brings one past the maps back to the last square. The same for `bottom` with `h - 1`, and for `left` and `right` with `w - 1`.",
            "`return F.adaptive_max_pool2d(maps[:, r0:r1 + ___, c0:c1 + ___], k)`.",
        ],
        why="For the first box, `(5.5, 9.0, 30.2, 44.0)`: rows `5.5 // 8` = 0 to `30.2 // 8` = 3, and columns 1 to 5, a region of 4 by 5 squares. 2 by 2 bins take rows 0–1 and 2–3, and columns 0–2 and 2–4 of it: the middle column is in both. For the box that sticks out, `(-6.0, 30.0, 20.0, 60.0)`: `-6.0 // 8` is −1, and without `max` the slice `-1:3` would be empty.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proposals():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The proposals

        Complete `proposals(out, anchors, n)`, step 2 of the recipe: `out` holds the rows of the RPN for one scene, a tensor of shape (A, 6), for every anchor 2 logits, background and object, and 4 offsets; `anchors` has shape (A, 4). It gives back the boxes of the first `n` proposals, a tensor of at most `n` rows of 4, fewer when NMS keeps fewer, the surest first: the probability of an object of every anchor, its box from `boxes_from` of your project, the 100 anchors with the highest probability, `nms` on their boxes at 0.7, and the boxes of the first `n` that it keeps. `nms` gives places in the boxes you give it. The check tries six made-up anchors, of which NMS keeps five, and the 768 anchors of a scene with random rows, as an RPN that has not learned yet. When it is right, it saves `proposals` in your project, as `work/projects/cnn/proposals.py`.
        """),
        L.task("Make `proposals` give the boxes of the first n proposals."),
    ])
    return


@app.cell(hide_code=True)
def exercise_proposals():
    box_proposals = L.code_box("proposals", lines=9)
    box_proposals
    return (box_proposals,)


@app.cell(hide_code=True)
def check_proposals(box_proposals):
    result_proposals = L.check("proposals", box_proposals.value)
    result_proposals
    return (result_proposals,)


@app.cell(hide_code=True)
def hints_proposals():
    L.hints(
        "proposals",
        [
            "Step 2 of the worked example, as a function, with `boxes_from(anchors, out[:, 2:])` for the written-out boxes and `n` for 32.",
            "`e = torch.exp(out[:, :2])` and `score = e[:, 1] / e.sum(dim=1)`; then `best = score.argsort(descending=True)[:100]`.",
            "`kept = nms(boxes[best], score[best], 0.7)[:n]`: places in `boxes[best]`. So `return boxes[___][kept]`.",
        ],
        why="Of the six anchors, anchor 1 is the surest, 0.953, and anchor 0, 0.881, overlaps it by 0.78: NMS drops it. Anchors 4 and 5 overlap by 0.6, below 0.7: both stay. So with `n` of 6 you get 5 boxes: anchors 1, 3, 2, 4 and 5, anchor 3 moved 0.25 × 16 = 4 pixels down by its offset.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_learns():
    mo.md(r"""
    ### Exercise 4. What a proposal learns
    """)
    return


@app.cell(hide_code=True)
def choice_learns_cell():
    choice_learns = L.choice(
        "learns",
        question="In training, a proposal overlaps the circle with an IoU of 0.62 and the square with 0.30. What does the second stage learn from it?",
        options={
            "right": "Class 1, circle, and the offsets from the proposal to the circle's box",
            "anchor": "Class 1, circle, and the offsets from the anchor it came from to the circle's box",
            "background": "Background: the RPN already found the circle, and only the true boxes learn a kind",
            "ignored": "Nothing: it lies partly on two shapes, so it is left out",
        },
        correct="right",
        why={
            "right": "Its best IoU, 0.62, is 0.5 or more, with the circle: class 0 + 1 = 1, and the offsets that turn the proposal into the circle's box. The second stage starts from the proposal, so that is what its offsets correct.",
            "anchor": "The offsets of the second stage start from the proposal: at detecting, `boxes_from(proposals, offsets)` makes the boxes. The anchor already did its work in the first stage.",
            "background": "Every proposal with an IoU of 0.5 or more with a true box learns that box's kind; the true boxes only join them. Correcting a proposal of 0.62 is just what the second stage must learn.",
            "ignored": "The second stage ignores nothing: from 0.5 on a kind, below 0.5 background. Its best IoU, 0.62, is with the circle.",
        },
    )
    choice_learns
    return (choice_learns,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains the course's two-stage detector, the backbone of your one-stage detector, an RPN and a second stage, with your `proposals` and your `roi_pool`, for 8 epochs on the 2,000 training scenes, in mini-batches of 32, as your one-stage detector trained. Then it finds the shapes of the 200 test scenes with them and your `boxes_from`. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; that takes a few minutes. While it trains, the chart shows the training loss of both stages together every 10 batches. Under the chart are the boxes the detector gives three test scenes at that moment, and under them the feature maps of the last layer of its backbone, which both stages read, on the scene of the worked example; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train the two-stage detector")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_two_stages(train_button, result_roi, result_proposals):
    result_roi, result_proposals  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the two-stage detector.*")
    else:
        _now = {}   # the live chart, made at the first report
        _words = {"scene": ("scene {seed}: {n} box", "scene {seed}: {n} boxes"), "layer": "Layer {name} of the detector on scene {seed}: the maps both stages read",
                  "alt": "Three test scenes, each with its true boxes and the dashed boxes of 0.5 or more that the detector gave it after that epoch, after NMS."}
        _words_now = _words | {"alt": "Three test scenes, each with its true boxes and the dashed boxes of 0.5 or more that the detector gives it now, after NMS."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Two stages: the training loss", "x": "epoch", "y": "training loss",
                                        "alt": "The training loss of the two-stage detector so far, both stages together, every 10 batches."}, _p["epochs"], low=0.0)
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-rcnn-now"))
            if "rpn_loss" in _p:   # after every epoch: its mean
                _loss = _p["rpn_loss"] + _p["second_loss"]
                _now["chart"].line(f"**Two stages** · epoch {_p['epoch']} of {_p['epochs']} · training loss {L.number(_loss, 3)}")
            else:   # every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Two stages** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · training loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `roi_pool` and your `proposals` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than fifteen minutes. Does your `roi_pool` or your `proposals` do more work than it needs, such as a loop over the anchors or over the bins? `F.adaptive_max_pool2d` cuts all bins in one step.*",
                "roi_pool": "*Your `roi_pool` in `work/projects/cnn/roi_pool.py` stops with an error, or does not give 64 × 4 × 4 numbers that training can follow back. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "proposals": "*Your `proposals` in `work/projects/cnn/proposals.py` stops with an error, or does not give between 1 and 32 boxes for a scene. Did you change its file? Delete it, and check exercise 3 again.*",
                "offsets": "*Your `boxes_from` in `work/projects/cnn/offsets.py` stops with an error on the detector's offsets. Did you change its file? Delete it, and check its exercise in [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss of each stage", "x": "epoch", "y": "loss", "rpn": "first stage", "second": "second stage",
                       "alt": "The training loss of each stage after every epoch: of the first stage solid, of the second stage dashed."}
            _curves, _moments = _result["curves"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"After epoch {_e}: loss {L.number(_curves['rpn'][_e - 1], 3)} for the first stage and {L.number(_curves['second'][_e - 1], 3)} for the second, and {_n} {'box' if _n == 1 else 'boxes'} of 0.5 or more on these three scenes, which have {C.SHOWN_SHAPES} shapes.")

            if len(_moments) == len(_curves["rpn"]) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_moments), values=list(range(1, len(_moments) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: after which epoch do the boxes sit on the shapes?")]
            else:
                _pic = [mo.Html(C.curves_picture(_curves, _labels, L.lang))] if _curves["rpn"] else []
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("circles", "squares", "triangles"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""The two-stage detector, with {C.whole(_result['params'], L.lang)} parameters, finds {_result['found']} of the {_result['shapes']} shapes of the 200 test scenes: a box of their kind, with a score of 0.5 or more and an IoU above 0.5. {_result['false']} of its boxes of 0.5 or more find no shape, and {_result['covered']} shapes have a proposal with an IoU above 0.5. Its mAP at an IoU of 0.5 is {L.number(_result['map'], 3)}, with an AP of {_aps}; at 0.75, it is {L.number(_result['map75'], 3)}. The course's two-stage detector, trained the same way with the seeds 0 to 4, finds 761 to 776 shapes, with an mAP of 0.947 to 0.962, and 0.734 to 0.779 at 0.75. The course's one-stage detector, trained the same way with the seeds 0 to 4, found 725 to 749 shapes, with an mAP of 0.906 to 0.918, and 0.807 to 0.822 at 0.75.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_second():
    mo.md(r"""
    ### Exercise 5. What the second stage sees
    """)
    return


@app.cell(hide_code=True)
def explain_second_cell():
    explain_second = L.explain(
        "why-second",
        question="Your one-stage detector and the two-stage detector have the same backbone. On the course's scenes, the two-stage detector finds more shapes, but at an IoU of 0.75 its boxes fit less well. What does its second stage see of a shape that the head of your one-stage detector does not? And what in RoI pooling could make its boxes fit less well?",
        model_answer="The head of the one-stage detector is one convolution of 3 by 3: it decides about every anchor from the same window of 3 by 3 squares around the anchor's square, of a fixed place and size, whatever the shape. The second stage gets the features of the proposal itself, cut to 4 by 4 bins whatever its size, and has its own linear layers to decide with: it judges a proposal that the RPN only thought likely to hold something, from the features of just that region. That costs a second pass for every proposal and many more parameters, 167,386 against 38,616. RoI pooling rounds a proposal to whole squares of 8 pixels: proposals whose corners lie in the same squares get the same numbers, so the second stage cannot see how far into a square a box begins, and its offsets cannot place the edges as exactly. RoIAlign, which does not round, made the boxes fit much better at 0.75.",
        points=[
            "you say that the head of the one-stage detector decides about an anchor from a fixed 3 by 3 squares around it, while the second stage gets the features of the whole proposal, pooled to the same size, with layers of its own",
            "you say that this costs something: a second pass for every proposal, and more parameters",
            "you say that RoI pooling rounds a proposal to whole squares, so proposals that differ by a few pixels get the same numbers, and the second stage cannot place the edges of a box exactly",
        ],
    )
    explain_second
    return (explain_second,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The anchors, targets and loss of [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) are now the first stage: an RPN that tells *object* from *background* and proposes boxes. Your `proposals` keeps the surest of them, with NMS at 0.7; your `roi_pool` cuts each from the feature maps to 4 by 4 bins; and a second stage gives each a kind and offsets from the proposal, followed by the NMS of [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en) at 0.5. That is the path from R-CNN, which ran a CNN on every proposal, through Fast R-CNN, which ran it once, to Faster R-CNN, whose proposals come from the network itself. On the course's scenes, the two-stage detector finds more shapes than your one-stage detector, but RoI pooling rounds, and at an IoU of 0.75 its boxes fit less well. Your project `cnn` now has `roi_pool.py` and `proposals.py`. In the next lesson, *A feature pyramid for small and large objects*, a detector gets feature maps of several sizes, so that it finds small objects too. Later, focal loss lets RetinaNet, a one-stage detector, match the two-stage detectors, and Mask R-CNN gives every box a mask and replaces RoI pooling with RoIAlign.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fast, result_roi, result_proposals, choice_learns, explain_second):
    L.wrap_up(choice_fast, result_roi, result_proposals, choice_learns, explain_second)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
