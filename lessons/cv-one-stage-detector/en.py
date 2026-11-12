# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-one-stage-detector. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-one-stage-detector", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_learn():
    quick_learn = L.choice(
        "quick-learn",
        question="An anchor has an IoU of 0.46 with box A and 0.31 with box B, and it is the best anchor of neither box. What does it learn?",
        options={
            "box": "Box A, the box it overlaps most",
            "background": "Background: it reaches 0.5 with no box",
            "right": "Nothing: it is ignored",
        },
        correct="right",
        why={
            "box": "An anchor learns the box it overlaps most only from an IoU of 0.5 on, or when it is that box's best anchor. At 0.46 it lies partly on A. See [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en).",
            "background": "Background is below 0.4. From 0.4 up to 0.5, an anchor lies partly on a shape, and is ignored. See [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en).",
            "right": "Its best IoU, 0.46, lies from 0.4 up to 0.5: it learns neither A nor background. Today, such an anchor gets class -1, and the loss leaves it out.",
        },
    )
    quick_learn
    return (quick_learn,)


@app.cell(hide_code=True)
def quick_check_back():
    quick_back = L.choice(
        "quick-back",
        question="An anchor of 16 by 16 pixels, with its middle at row 43.5 and column 59.5, gets the box offset (0.25, 0, log 2, 0). Which box `(top, left, bottom, right)` does that make?",
        options={
            "right": "(32, 52, 63, 67): 4 pixels lower, twice as high",
            "pixels": "(28.25, 52, 59.25, 67): a quarter of a pixel lower, twice as high",
            "wide": "(40, 44, 55, 75): 4 pixels lower, twice as wide",
        },
        correct="right",
        why={
            "right": "The middle moves 0.25 × 16 = 4 rows down, to 47.5, and the height becomes 16 × e^(log 2) = 32: rows 47.5 − 15.5 = 32 to 63. The width stays 16: columns 52 to 67. Your `boxes_from` does this for every anchor at once, and today it turns your detector's offsets into boxes.",
            "pixels": "t_y is a share of the anchor's height, not pixels: 0.25 × 16 = 4 rows down. See [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en).",
            "wide": "The box offset goes rows first: (t_y, t_x, t_h, t_w). The third number, log 2, is the change of height. See [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en).",
        },
    )
    quick_back
    return (quick_back,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en), you ran DETR, a detector that others trained on COCO. In [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en), you told every anchor of a scene what to learn. Now you build a detector and train it yourself: one CNN that reads a scene once, and gives every anchor at once a kind and a box offset. A detector that works this way is a **one-stage detector**. SSD (Liu and colleagues, 2016) and YOLOv2 (Redmon and Farhadi, 2017) are two well-known ones, and RetinaNet a later one. A two-stage detector, such as Faster R-CNN, first picks regions that may hold an object, and then looks at each of them again: that is the next lesson. On the course's scenes, your detector trains in a few minutes on a CPU, and finds most of the shapes of scenes it never saw.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working", "Your CNN on color photos", "Your own one-stage detector",
                            "Your own U-Net", "Final project: ResNet with Grad-CAM", "Final project: U-Net or detector"],
                           5, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_found():
    guess_found = L.guess(
        "found",
        question="Your detector has 38,616 parameters. It trains for 8 epochs on 2,000 scenes, in a few minutes on a CPU. Then it gets 200 scenes it never saw, with 808 shapes. How many of them does it find, with a box of the right kind, a score of 0.5 or more, and an IoU above 0.5 with the shape?",
        options={
            "half": "About 400: half of them",
            "most": "About 600: three in four",
            "right": "About 730: nine in ten",
            "all": "All 808",
        },
        reveal="About 730. The course's detector, trained as the demo trains yours, with the seeds 0 to 4, finds 725 to 749 of the 808 shapes. Small shapes it misses far more often: with seed 0, it misses 25 of the 95 shapes whose box is at most 20 pixels on its longest side, against 32 of the 550 of 25 pixels or more. A small shape has few anchors that learn it: in the last lesson, the small triangle had 2, the circle 8.",
    )
    guess_found
    return (guess_found,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One pass over the scene.** The first part of your detector, the **backbone**, is a CNN like your earlier ones, now with batch normalization after every convolution, as ResNet has it: convolutions of 3 by 3, each followed by batch normalization and ReLU, and three poolings of 2. They make a scene of 128 by 128 pixels into 64 feature maps of 16 by 16 squares, each square standing for 8 by 8 pixels: the squares of the anchors. Then comes the head, the part that gives the answers, as the last layer of your CNNs gave their logits: one convolution of 3 by 3, with 24 output channels. For each of the 3 anchors of a square, it gives 8 numbers: 4 logits, for background, circle, square and triangle, and 4 box offsets. The same filters slide over every square, so every square answers the same question for its own anchors, wherever a shape lies. One pass gives 16 × 16 × 24 = 6,144 numbers: 8 for each of the 768 anchors. YOLOv2 too gives the anchors of one feature map their numbers, 5 on each of 13 by 13 squares; SSD does it with convolutions of 3 by 3 on several feature maps of different sizes.

        **In the order of the anchors.** The head gives a tensor of shape (N, 24, 16, 16): the channels first, then the rows and the columns. `permute(0, 2, 3, 1)` puts the channels last, (N, 16, 16, 24), so that the 24 numbers of a square lie together; `reshape(N, 768, 8)` then cuts them into 3 rows of 8, square by square, row by row. That is the order of `anchor_boxes`: row 458 belongs to anchor 458.

        **What each anchor learns.** Its class: 0 for background, and the kind of its box plus 1 for an anchor that learns a box, so 1 for a circle, 2 for a square and 3 for a triangle; -1 when it is ignored. Background is class 0 here, where DETR put its *no object* last. And for an anchor that learns a box, the box offset from your `offsets`.

        **Two losses in one.** One loss for the kinds: the cross-entropy of the 4 logits, over every anchor that is not ignored, 755 of the 768 on the scene of the worked example. One loss for the boxes: the mean of the absolute differences between the predicted and the true offsets, the mean absolute error of [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en), but only over the 20 anchors that learn a box: a background anchor has no box to learn. The detector is trained on their sum. SSD adds two such losses too, with a weight of 1, but it adds each up over its anchors and divides by the number of anchors that learn a box, and for the boxes it takes smooth L1, a variant of the absolute difference.

        **From 768 rows to boxes.** Softmax turns the 4 logits of an anchor into probabilities. Its **score** is the highest probability of a kind, leaving background out, as DETR's score left out *no object*. An anchor with a score of 0.5 or more keeps its box: your `boxes_from` makes it from the anchor and its offsets. Last comes the non-maximum suppression of [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en), for each kind on its own, at 0.5. For the mAP of [Score a detector: mean average precision](/course/open/cv-map?lang=en), the boxes count down to a score of 0.05.

        **Why NMS.** In training, every shape is learned by several anchors, each with its own offsets: on the scene of the worked example, 8 anchors learn the circle. So the trained detector gives one shape several boxes: on the 200 test scenes, about 20 boxes of 0.5 or more per scene before NMS and 4.3 after, for 4 shapes per scene on average.
        """),
        L.watch_out("Background outnumbers everything. On the scene of the worked example, 735 of the 755 anchors in the kind loss learn background: a detector that said background everywhere would be right for 97% of them. On these scenes, your detector learns the shapes all the same. SSD keeps only the hardest background anchors, at most 3 for every anchor that learns a shape; RetinaNet changes the loss itself, in a later lesson."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **one-stage detector** | a detector whose one network gives, in one pass, every anchor of the image a kind and a box offset; NMS then keeps the best boxes. SSD, YOLOv2 and RetinaNet are one-stage detectors | `Detector` |
        | **backbone** | the part of a network that turns an image into feature maps; a few layers on top, the head, turn those into the answers for the task. Here a CNN of 4 convolutions | `self.backbone` |
        | **head** | the layers on top of the backbone that give the answers; here one convolution of 3 by 3 that gives every anchor its 4 logits and 4 offsets | `self.head` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_nms():
    mo.vstack([
        mo.md("**The course's trained detector on the scene of the worked example**"),
        mo.Html(C.nms_picture({
            "before": "before NMS: {n} boxes of 0.5 or more",
            "after": "after NMS: {n} boxes",
            "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
            "alt": "The scene of the worked example twice, with its four true boxes, solid. Left, the 17 dashed boxes of the anchors with a score of 0.5 or more: 7 on the circle, 5 on the large triangle, and 5 on the square, 2 of which say triangle; none on the small triangle. Right, the 4 boxes that NMS keeps, each with its kind and score: circle 0.99, triangle 0.94, triangle 0.75 and square 0.74.",
        })),
        mo.md(r"""
        On the left, 17 anchors have a score of 0.5 or more: 7 around the circle, 5 around the large triangle and 5 around the square. NMS, for each kind on its own, keeps the box with the highest score, and drops the boxes of the same kind that overlap it by more than 0.5. On the right, 4 boxes stay. Three are right: the circle, 0.99, the large triangle, 0.94, and the square, 0.74. The fourth, a triangle with 0.75, lies on the square, where the circle covers part of it: NMS keeps it, because it is the best triangle there, and no box of another kind removes it. The small triangle, 16 by 16 pixels, gets no box of 0.5 or more.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How hard should the box loss pull? Both losses train the same backbone. The slider multiplies the box loss by a weight before it is added: from 0, no box loss, to 16. Every stop is the course's detector, trained as in the demo with seed 0, but with that weight. On the left, its boxes of 0.5 or more on the scene of the worked example, after NMS; on the right, its mAP on the 200 test scenes, at an IoU of 0.5, solid, and of 0.75, dashed: there a box must fit much better to count.
    """)
    return


@app.cell(hide_code=True)
def explore_weight_cell():
    def _frame(w):
        w = float(w)
        _html, _stop = C.weight_frame(w, {"title": "weight {weight}: {n} boxes", "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
                                          "chart": "mAP on the 200 test scenes", "weight": "weight of the box loss",
                                          "at50": "IoU 0.5", "at75": "IoU 0.75",
                                          "chart_alt": "The mAP of the course's detector on the 200 test scenes at each weight of the box loss, at an IoU of 0.5 and of 0.75: 0.005 and 0 at weight 0, the highest at 0.25, 0.947 and 0.861, then 0.916 and 0.819 at 1, 0.790 and 0.712 at 4, and 0.675 and 0.598 at 16.",
                                          "alt": "The scene of the worked example with its four true boxes and the {n} dashed boxes of 0.5 or more of the detector trained with weight {weight} on its box loss. Next to it, its mAP at every weight."})
        return _html, f"Weight {C.weight_text(w, '.')}: mAP {L.number(_stop['map'], 3)} at an IoU of 0.5 and {L.number(_stop['map75'], 3)} at 0.75, and {len(_stop['boxes'])} boxes on this scene."

    explore_weight = L.explore("weight-slider", "weight of the box loss", _frame, start=1.0, values=list(C.WEIGHTS), role="hyper",
                               task="Find the weight with the highest mAP at an IoU of 0.75. What happens to the boxes at 0, and to the kinds at 16?")
    explore_weight
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        At 0, the box loss is gone, and the offsets never train. The detector still learns which anchors lie on which kind of shape, but its boxes land anywhere: 21 boxes of 0.5 or more on this scene after NMS, and an mAP of 0.005. At 0.25, the mAP is the highest, 0.947 at an IoU of 0.5 and 0.861 at 0.75, and the small triangle gets a box too. At 4 and 16, the box loss outweighs the loss of the kinds, and the detector grows unsure of them: at 4, one box calls the circle a square, with 0.60, and another a circle, with only 0.51; at 16, one box keeps a score of 0.5 or more, on the large triangle. The mAP falls to 0.790 and 0.675.
        """),
        L.watch_out("Do not choose the weight from this chart: it measures the test scenes. A weight chosen on them makes their mAP promise more than new scenes would give. Choose it on a validation set, scenes you split off from the training scenes, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en). This lesson keeps a weight of 1, as SSD does, though SSD's makers chose it on their own data, for their own way of adding the two losses."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        A one-stage detector:

        1. Lay the anchors: around the middle of every square of the feature map, one of each size. On the course's scenes, `anchor_boxes()` gives 768.
        2. For every scene, what every anchor learns: its class, 0 for background, the kind + 1 for an anchor that learns a box, -1 when it is ignored; and for an anchor that learns a box, its box offset.
        3. The network: a backbone that makes the scene into feature maps with one square per square of the anchors, and a head, a convolution of 3 by 3 with (kinds + 1 + 4) channels for every anchor of a square. `permute` and `reshape` make that one row per anchor, in the order of the anchors.
        4. The loss: the cross-entropy of the logits over the anchors that are not ignored, plus the mean absolute difference of the offsets over the anchors that learn a box.
        5. Train it on many scenes.
        6. Detect: softmax over the logits of every anchor, its score the highest probability of a kind; keep the anchors with a score of 0.5 or more, make their boxes with `boxes_from`, and run NMS for every kind.
        7. Measure it on scenes it never saw: the mAP, with the boxes down to a score of 0.05.
        """),
        L.watch_out("Each loss leaves out its own anchors. The kind loss leaves out the ignored anchors, class -1: `F.cross_entropy` stops on a class of -1. The box loss leaves out the background anchors too, class 0: they have no box, and their offsets would be pulled to 0 for nothing."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example does steps 1 to 4 and the start of step 6 on the scene of the worked example of the last lesson, `scene(18)`, with a network that has not trained yet. Step 2 uses `match_anchors` from `everything_ai.scenes`, which gives what your `match` gave in the last lesson, for every anchor: the number of its box, -1 for background, -2 when it is ignored. The offsets in step 2 are the PyTorch cell of the last lesson. In step 3, the backbone is an `nn.Sequential`: it runs its layers in order, as a `forward` would. `nn.BatchNorm2d(8)` is the batch normalization of [Batch normalization](/course/open/nn-batchnorm?lang=en) for maps: each of the 8 channels gets its own mean and standard deviation, over the batch and every square. `nn.ReLU()` is `torch.relu` as a layer. In step 6, `.max(dim=1)` gives two things for every row: its highest number, and where it lies, as `argmax(dim=1)` does; `kind` is then the best kind of every anchor, 0 for a circle, which NMS for every kind needs. The rest of step 6, your `boxes_from` and the NMS of [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en), and steps 5 and 7 take longer: the demo after the exercises does them with your code. Run the cell as it is; then take out the `permute(0, 2, 3, 1)` where the comment says, and run it again.
    """)
    return


@app.cell
def example_detector():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from everything_ai.scenes import anchor_boxes, match_anchors, scene

    s = scene(18)
    truth = torch.tensor([shape.box for shape in s.shapes]).float()   # the true boxes, one row per shape
    kinds = torch.tensor([shape.label for shape in s.shapes])          # 0 circle, 1 square, 2 triangle
    print(kinds)

    # 1. the anchors, as in the last lesson: 768, row by row, square by square, size by size
    anchors = torch.tensor(anchor_boxes()).float()
    # 2. what each anchor learns: its class, 0 for background, the kind + 1 for a shape, -1 when
    #    it is ignored; and the box offset of every anchor that learns a box
    label = torch.tensor(match_anchors(anchors, truth))   # as your match: a box, -1 or -2
    learn = label >= 0
    classes = torch.zeros(768, dtype=torch.int64)
    classes[learn] = kinds[label[learn]] + 1
    classes[label == -2] = -1
    a, b = anchors[learn], truth[label[learn]]
    middle_a, size_a = (a[:, :2] + a[:, 2:]) / 2, a[:, 2:] - a[:, :2] + 1
    middle, size = (b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2] + 1
    t = torch.zeros(768, 4)
    t[learn] = torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)
    print((classes == 0).sum(), (classes > 0).sum(), (classes == -1).sum())
    print(classes[458], t[458])

    # 3. the network: three poolings make 16 x 16 squares of 8 pixels, and the head
    #    gives every square 3 x 8 numbers: 4 logits and 4 offsets for each of its anchors
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    )
    head = nn.Conv2d(64, 3 * 8, 3, padding=1)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        maps = head(backbone(x))
        out = maps.permute(0, 2, 3, 1).reshape(1, 768, 8)   # change: maps.reshape(1, 768, 8)
        print(x.shape, maps.shape, out.shape)
        print(out[0, 458])
        print(maps[0, 16:24, 9, 8])   # the 8 numbers of the third anchor of square (9, 8)

        # 4. the loss: the cross-entropy over the anchors that are not ignored, plus the
        #    mean absolute difference of the offsets of the anchors that learn a box
        kept = classes >= 0
        kind_loss = F.cross_entropy(out[0, kept, :4], classes[kept])
        box_loss = (out[0, learn, 4:] - t[learn]).abs().mean()
        print(kept.sum(), kind_loss, box_loss)

        # 6. detect: softmax over the 4 logits of every anchor; its score is the highest
        #    probability of a kind
        e = torch.exp(out[0, :, :4])
        p = e / e.sum(dim=1, keepdim=True)
        score, kind = p[:, 1:].max(dim=1)
        print(p[458], score.max(), (score >= 0.5).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The scene has two triangles, a square and a circle: kinds 2, 2, 1 and 0. As in the last lesson, 735 anchors learn background, 20 a shape and 13 are ignored, and anchor 458, the circle's best, gets class 1 and the offsets −0.1094, twice, and −0.0984, twice. The scene has shape (1, 3, 128, 128), the head's maps (1, 24, 16, 16), and the rows (1, 768, 8). Row 458 holds the same 8 numbers as channels 16 to 23 of square (9, 8): anchor 458 is (9 × 16 + 8) × 3 + 2, the third anchor of that square.

    The kind loss over the 755 anchors that are not ignored is 1.4667, close to log 4 = 1.386: the loss of a network that gives all 4 classes the same probability. The box loss over the 20 anchors that learn a box is 0.6996. Untrained, anchor 458 gives background 0.47 and circle 0.16, but elsewhere 16 anchors already have a score of 0.5 or more, the highest 0.573: training has to teach the detector background there.

    With the change, the shape stays (1, 768, 8), but row 458 becomes (0.1454, 0.2703, …): 8 numbers of channel 14, from squares (5, 0) to (5, 7), and no longer the 8 numbers of anchor 458.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_head():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The size of the head
    """)
    return


@app.cell(hide_code=True)
def choice_head_cell():
    choice_head = L.choice(
        "head-size",
        question="A one-stage detector for the 20 kinds of PASCAL VOC has 9 anchors on every square, as Faster R-CNN. Its head is a convolution of 3 by 3, as yours. How many output channels does it need?",
        options={
            "no-background": "216: 9 anchors times 24, for 20 kinds and 4 offsets",
            "right": "225: 9 anchors times 25, for 20 kinds, background and 4 offsets",
            "one": "25: 20 kinds, background and 4 offsets, for the square",
            "no-offsets": "189: 9 anchors times 21, for 20 kinds and background",
        },
        correct="right",
        why={
            "no-background": "Every anchor also needs a logit for background: almost every anchor learns it. 20 kinds + background + 4 offsets is 25 per anchor.",
            "right": "Every anchor of a square gets its own 25 numbers: 21 logits, the 20 kinds and background, and 4 offsets. 9 × 25 = 225 channels. Yours has 3 × (3 + 1 + 4) = 24.",
            "one": "That is one anchor per square. Each of the 9 anchors gets its own kind and its own box offset: 9 × 25.",
            "no-offsets": "Without offsets, every box stays its anchor. Each anchor also needs its 4 offsets: 9 × 25.",
        },
    )
    choice_head
    return (choice_head,)


@app.cell(hide_code=True)
def text_exercise_targets():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What every anchor learns

        Complete `targets(anchors, boxes, labels)`, step 2 of the recipe: it gets the anchors, a tensor of shape (A, 4), the true boxes of a scene, a tensor of shape (B, 4), and the kind of each box, a tensor of B whole numbers, 0 for a circle, 1 for a square and 2 for a triangle. It gives back `classes`, a tensor of A whole numbers, and `t`, a tensor of shape (A, 4): the worked example's step 2, as a function. The first line is given: it takes `iou_matrix` and `match` from your project, saved in the last lesson, and makes what `match` gives into a tensor, `label`. `offsets` comes from your project too. The check gives your function four anchors and two boxes, the scene of the worked example, and another scene with other anchors. It compares the classes, and the offsets of the anchors that learn a box. When it is right, it saves `targets` in your project, as `work/projects/cnn/targets.py`: the demo makes the targets of every scene with it.
        """),
        L.task("Make `targets` give every anchor its class and its box offset."),
    ])
    return


@app.cell(hide_code=True)
def exercise_targets():
    box_targets = L.code_box("targets", lines=12)
    box_targets
    return (box_targets,)


@app.cell(hide_code=True)
def check_targets(box_targets):
    result_targets = L.check("targets", box_targets.value)
    result_targets
    return (result_targets,)


@app.cell(hide_code=True)
def hints_targets():
    L.hints(
        "targets",
        [
            "Step 2 of the worked example, with `labels` for `kinds` and `len(anchors)` for 768: a mask of the anchors that learn a box, classes that are all 0, of whole numbers, then the kinds, and -1 for the ignored anchors.",
            "`learn = label >= 0` marks the anchors that learn a box, and `labels[label[learn]]` is the kind of the box of every one of them: an array as an index. Add 1 to it, and give -1 where `label == ___`.",
            "The offsets: zeros of shape (A, 4), then `t[learn] = offsets(anchors[learn], boxes[___])`. End with `return classes, t`.",
        ],
        why="For the four anchors: anchor 0 learns the triangle, kind 2, class 3, with the offsets (0, 0.125, 0, 0): the triangle's middle lies 2 pixels to the right, an eighth of 16. Anchor 1 reaches 0.45 with it: ignored, -1. Anchor 2 learns the circle, class 1, its middle 2 pixels lower: 2 / 32 = 0.0625. Anchor 3 learns background, 0. So `classes` is `[3, -1, 1, 0]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. One loss for kinds and boxes

        Complete `detection_loss(out, classes, t)`, step 4 of the recipe: `out` is what the detector gives for a batch of N scenes, a tensor of shape (N, A, 8), for every anchor 4 logits and 4 offsets; `classes`, of shape (N, A), and `t`, of shape (N, A, 4), are what `targets` gives for each scene, put together. It gives back one number: the cross-entropy of the logits `out[:, :, :4]` over the anchors with a class of 0 or more, plus the mean of the absolute differences between the offsets `out[:, :, 4:]` and `t` over the anchors with a class above 0. A mask of shape (N, A), such as `classes >= 0`, keeps those anchors of every scene at once: `out[:, :, :4][kept]` has shape (K, 4). The check tries one scene of the four anchors of exercise 2, and two made-up batches of two scenes. When it is right, it saves `detection_loss` in your project, as `work/projects/cnn/detection_loss.py`.
        """),
        L.task("Make `detection_loss` add the loss of the kinds and the loss of the boxes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_detection_loss = L.code_box("detection-loss", lines=8)
    box_detection_loss
    return (box_detection_loss,)


@app.cell(hide_code=True)
def check_loss(box_detection_loss):
    result_loss = L.check("detection-loss", box_detection_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "detection-loss",
        [
            "Step 4 of the worked example, for a batch: a mask has shape (N, A) now, and `out[:, :, :4][kept]` keeps the logits of those anchors of every scene, one row each.",
            "The kind loss takes the anchors of `kept = classes >= 0`: `F.cross_entropy(out[:, :, :4][kept], classes[___])`. The box loss takes another mask: `learn = classes > ___`.",
            "`(out[:, :, 4:][learn] - t[learn]).___().mean()`, and give back the sum of the two losses.",
        ],
        why="For the scene of four anchors: the cross-entropy of anchors 0, 2 and 3, whose classes are 3, 1 and 0, is about 0.994; the offsets of anchors 0 and 2 differ by 0.1, 0.075, 0, 0 and 0.0625, 0, 0.1, 0, which makes a mean of about 0.042. Together, about 1.036. Anchor 1, class -1, takes part in neither.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_detector():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Your detector

        `Detector` holds the backbone of the worked example. Complete it, step 3 of the recipe: make `self.head` a convolution of 3 by 3 from the 64 channels of the backbone to 3 × 8 channels, with `padding=1`; and make `forward` give, for a batch `x` of shape (N, 3, 128, 128), a tensor of shape (N, 768, 8), one row per anchor, in the order of `anchor_boxes`. The check makes networks after `torch.manual_seed`, gives them random scenes, and compares the shape, the rows of a few anchors, and the number of parameters. When it is right, it saves `Detector` in your project, as `work/projects/cnn/detector.py`.
        """),
        L.task("Make `Detector` give every anchor its 4 logits and 4 offsets."),
    ])
    return


@app.cell(hide_code=True)
def exercise_detector():
    box_detector = L.code_box("detector", lines=18)
    box_detector
    return (box_detector,)


@app.cell(hide_code=True)
def check_detector(box_detector):
    result_detector = L.check("detector", box_detector.value)
    result_detector
    return (result_detector,)


@app.cell(hide_code=True)
def hints_detector():
    L.hints(
        "detector",
        [
            "Step 3 of the worked example, as a class: the head is made in `__init__`, after the backbone, and `forward` runs the scenes through both.",
            "`self.head = nn.Conv2d(64, ___, 3, padding=1)`: 3 anchors of 8 numbers each. In `forward`, the head gives (N, 24, 16, 16): first put the channels last.",
            "`out = self.head(self.backbone(x))`, then `return out.permute(0, 2, 3, 1).reshape(len(x), ___, 8)`.",
        ],
        why="The head has 64 × 3 × 3 × 24 = 13,824 weights and 24 biases. With the backbone's 24,768 parameters, your detector has 38,616. `reshape(len(x), -1, 8)` works out the 768 itself: 16 × 16 × 24 / 8.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo makes the targets of the 2,000 training scenes and the 200 test scenes with your `targets`, trains your `Detector` with your `detection_loss` for 8 epochs, in mini-batches of 32, as the course's detector trained, and then finds the shapes of the test scenes with your `boxes_from` and NMS. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; that takes a few minutes. While it trains, the chart shows the loss on the training scenes every 10 batches, and dashed, the loss on the 200 test scenes after every epoch. Under the chart are the boxes your detector gives three test scenes at that moment, and under them the feature maps of its last layer before the head, on the scene of the worked example; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your detector")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_detector(train_button, result_targets, result_loss, result_detector):
    result_targets, result_loss, result_detector  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your detector.*")
    else:
        _now = {}   # the live chart, made at the first report
        _words = {"scene": ("scene {seed}: {n} box", "scene {seed}: {n} boxes"), "layer": "Layer {name} of your detector on scene {seed}",
                  "alt": "Three test scenes, each with its true boxes and the dashed boxes of 0.5 or more that your detector gave it after that epoch, after NMS."}
        _words_now = _words | {"alt": "Three test scenes, each with its true boxes and the dashed boxes of 0.5 or more that your detector gives it now, after NMS."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your detector: the loss while it trains", "x": "epoch", "y": "loss",
                                        "alt": "The loss of your detector so far: on the training scenes every 10 batches, and dashed on the 200 test scenes after every epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "training scenes, every 10 batches"), ("loss", True, "test scenes, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-one-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test scenes after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your detector** · epoch {_p['epoch']} of {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:   # the training scenes every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your detector** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2, 3 and 4 first: then your `targets`, your `detection_loss` and your `Detector` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Your project takes too long here: more than two minutes for the targets of the 2,000 training scenes, or more than ten minutes to train. Does your `targets`, your `Detector` or your `detection_loss` do more work than it needs, such as a loop over the anchors?*",
                "targets": "*Your `targets` in `work/projects/cnn/targets.py` stops with an error, or does not give a class and 4 offsets for each of the 768 anchors of a scene. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "loss": "*Your `detection_loss` in `work/projects/cnn/detection_loss.py` stops with an error, or does not give back one number as a tensor that training can follow back. Did you change its file? Delete it, and check exercise 3 again.*",
                "detector": "*Your `Detector` in `work/projects/cnn/detector.py` stops with an error, or does not give 768 rows of 8 for a scene. Did you change its file? Delete it, and check exercise 4 again.*",
                "offsets": "*Your `boxes_from` in `work/projects/cnn/offsets.py` stops with an error on your detector's offsets. Did you change its file? Delete it, and check its exercise in [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "The loss of your detector after every epoch: on the training scenes solid, on the test scenes dashed."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"After epoch {_e}: test loss {L.number(_val[_e - 1], 3)}, and {_n} {'box' if _n == 1 else 'boxes'} of 0.5 or more on these three scenes, which have {C.SHOWN_SHAPES} shapes.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: after which epoch do the boxes sit on the shapes, and which shapes stay without a box?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("circles", "squares", "triangles"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Your detector, with {C.whole(_result['params'], L.lang)} parameters, finds {_result['found']} of the {_result['shapes']} shapes of the 200 test scenes: a box of their kind, with a score of 0.5 or more and an IoU above 0.5. {_result['false']} of its boxes of 0.5 or more find no shape. Its mAP at an IoU of 0.5 is {L.number(_result['map'], 3)}, with an AP of {_aps}; at 0.75, it is {L.number(_result['map75'], 3)}. Before NMS, it gives {L.number(_result['before'], 1)} boxes of 0.5 or more per scene. The course's detector, trained the same way with the seeds 0 to 4, finds 725 to 749 shapes, with an mAP of 0.906 to 0.918.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_nms():
    mo.md(r"""
    ### Exercise 5. Why NMS?
    """)
    return


@app.cell(hide_code=True)
def explain_nms_cell():
    explain_nms = L.explain(
        "why-nms",
        question="Your detector needs NMS; DETR, in [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en), did not. What in the training of each makes that difference? And if your `targets` let only the best anchor of every true box learn it, and gave every other anchor background, what would you expect?",
        model_answer="In training, a shape is learned by every anchor that reaches an IoU of 0.5 with it, and by its best anchors, each with its own offsets to the same box: the circle of the worked example had 8. After training, they all give the shape a box with a high score, about 20 boxes of 0.5 or more per scene for 4 shapes, and NMS keeps the box with the highest score and drops the boxes of the same kind that overlap it by more than 0.5. DETR matched each true box with exactly one query while it trained, and every other query had to learn *no object*: it learned to give each object one box. With only the best anchor learning a shape, I would expect fewer boxes per shape, but not one for sure: the anchors next to it see almost the same pixels and now have to learn background, which is harder, and some of them can still give the shape a box, so NMS stays useful.",
        points=[
            "you say that in your detector's training several anchors around a shape learn that shape, so after training they all give it a box, and NMS keeps the best of them",
            "you say that DETR matched every true box with one query in training, so the other queries learned *no object*, and one box per object is what it learned",
            "you say that with only the best anchor learning a shape you would expect fewer boxes, but that its neighbours, which see almost the same pixels, would have to learn background, so a few boxes can stay and NMS is still useful",
        ],
    )
    explain_nms
    return (explain_nms,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The targets of [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) are now what a network learns. Your `targets` gives every anchor a class and a box offset; your `Detector` gives every anchor 4 logits and 4 offsets in one pass, with a backbone as in your earlier CNNs and a head that is one convolution of 3 by 3; and your `detection_loss` adds the cross-entropy of the kinds to the mean absolute difference of the boxes. Detecting takes softmax and a score, as for DETR in [Use a pretrained detector](/course/open/cv-pretrained-detector?lang=en), your `boxes_from`, and the NMS of [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en): many anchors learned each shape, so each shape gets many boxes. Trained for 8 epochs on 2,000 scenes, the course's detector finds 725 to 749 of the 808 shapes of 200 new scenes, with an mAP of 0.906 to 0.918; with seed 0, it misses small shapes far more often than large ones. Your project `cnn` now has `targets.py`, `detection_loss.py` and `detector.py`. In the next lesson, *From R-CNN to Faster R-CNN: two stages*, a detector first proposes regions that may hold an object, and then looks at each of them again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_head, result_targets, result_loss, result_detector, explain_nms):
    L.wrap_up(choice_head, result_targets, result_loss, result_detector, explain_nms)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
