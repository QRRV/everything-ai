# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-focal-loss. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-focal-loss", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A feature pyramid for small and large objects](/course/open/cv-fpn?lang=en) and [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hard():
    quick_hard = L.choice(
        "quick-hard",
        question="Hard negative mining, as your FPN detector was trained: in a scene where 10 of the 2,720 anchors learn a shape and none is ignored, the cross-entropy counts those 10 and the 30 background anchors with the highest cross-entropy. What happens to the cross-entropy of the other 2,680 background anchors?",
        options={
            "right": "It does not count in that batch at all, however sure the detector is of them",
            "third": "It counts for a third, so that background weighs as much as the shapes",
            "later": "It is kept, and counts in the next batch",
            "box": "It goes into the box loss",
        },
        correct="right",
        why={
            "right": "Hard negative mining is all or nothing: an anchor counts fully, or not at all. The other 2,680 background anchors get a weight of 0, also those the detector is only fairly sure of. Today's loss counts every anchor, each with a weight of its own.",
            "third": "Hard negative mining has no weights: the 40 anchors count fully, the others not at all. See [A feature pyramid for small and large objects](/course/open/cv-fpn?lang=en).",
            "later": "Every batch picks its own 30: the background anchors the detector gets most wrong at that moment. The others are not kept for later.",
            "box": "The box loss only counts the anchors that learn a shape: a background anchor has no box to learn. The other background anchors do not count at all.",
        },
    )
    quick_hard
    return (quick_hard,)


@app.cell(hide_code=True)
def quick_check_stages():
    quick_stages = L.choice(
        "quick-stages",
        question="The first stage of your two-stage detector gives each of the 768 anchors of a scene a probability of an object. How many boxes does its second stage look at again?",
        options={
            "right": "32 proposals: of the 100 anchors with the highest probability of an object, the first 32 that NMS at 0.7 keeps",
            "all": "All 768: the second stage looks at every anchor again",
            "hundred": "100: the anchors with the highest probability of an object",
            "shapes": "As many as the scene has shapes",
        },
        correct="right",
        why={
            "right": "From 768 anchors to 32 proposals: the first stage throws away almost all the background, so the second stage looks again at boxes that lie on or around a shape. A one-stage detector has no first stage: its loss sees every anchor.",
            "all": "Then the first stage would throw nothing away. It keeps the 100 most probable anchors, and NMS at 0.7 leaves the first 32 of them. See [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en).",
            "hundred": "Those 100 go into NMS at 0.7, and the first 32 boxes that stay are the proposals. See [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en).",
            "shapes": "The detector does not know how many shapes a scene has: it always passes on 32 proposals, and the second stage decides which of them hold a shape.",
        },
    )
    quick_stages
    return (quick_stages,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Until 2017, the most accurate detectors had two stages: the first throws away most of the background, so the second looks again at a few proposals, as your two-stage detector did with 32 per scene. A one-stage detector looks at every anchor at once, and almost all of them are background: in a scene of the course, about 10 of the 2,720 anchors learn a shape, and hard negative mining kept only the 30 hardest background anchors. The **focal loss** (Lin and colleagues, 2017) keeps all of them, but makes the loss of an anchor smaller the surer the detector already is; with it, **RetinaNet**, a one-stage detector with a feature pyramid, found the objects in the photos of COCO better than every two-stage detector before it. Today you train the course's FPN detector with it.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_easy():
    guess_easy = L.guess(
        "easy-share",
        question="The course trained this lesson's detector. After training, it gives almost every background anchor of a test scene a probability above 0.9 for background: about 2,690 of the 2,707 anchors of a scene that are not ignored, each with a cross-entropy below 0.11. About 10 anchors learn a shape. What share of the summed cross-entropy of a scene comes from those easy background anchors?",
        options={
            "none": "Almost none: each of them is tiny",
            "tenth": "About a tenth",
            "right": "More than half",
            "all": "Almost all of it",
        },
        reveal="56%: each of them is small, but there are about 2,690 of them, and the 10 anchors that learn a shape give 29%. That detector was trained with the focal loss of this lesson, which turns the loss of those easy anchors down: in its focal loss, they give 0.2%. The cross-entropy would go on making them surer, though the detector already gets them right.",
    )
    guess_easy
    return (guess_easy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Easy anchors add up.** Your one-stage detector took the cross-entropy of every anchor that is not ignored: $-\log p_t$, where $p_t$ is the probability that softmax gives the class the anchor learns, background or its kind. A background anchor the detector is sure of, at $p_t = 0.99$, has a cross-entropy of only 0.01. But a scene has about 2,700 of them, and about 10 anchors that learn a shape: together, the easy ones can outweigh the few that matter, as the guess showed. A two-stage detector throws most of the background away in its first stage; Faster R-CNN's RPN learns from 256 anchors per photo, at most half of them objects. A one-stage detector has no such stage. Hard negative mining, which you used in your FPN detector, counts only the hardest background anchors.

        **The focal loss.** The paper's answer: keep every anchor, and weight it by how wrong the detector still is,

        $$\mathrm{FL}(p_t) = -(1 - p_t)^{\gamma} \log p_t .$$

        With $\gamma = 0$, that is the cross-entropy. With $\gamma = 2$, the value of the paper, an anchor at $p_t = 0.9$ counts $(1 - 0.9)^2$, a hundredth, of its cross-entropy, and one at 0.99 a ten-thousandth; one at 0.5 still a quarter, and an anchor the detector gets wrong, at $p_t = 0.1$, 81%. $\gamma$ says how fast the weight falls: the larger it is, the sooner an anchor counts as done. In code, $p_t$ comes from the cross-entropy itself: the cross-entropy is $-\log p_t$, so $p_t = e^{-\text{CE}}$, `torch.exp(-ce)`. `F.cross_entropy(..., reduction="none")` gives the cross-entropy of every row, instead of their mean.

        **Summed, and divided by the shapes.** Your one-stage detector divided the cross-entropy by every anchor that is not ignored, about 2,700 in a scene of the FPN detector: its mean. RetinaNet sums the focal loss over all anchors, and divides it by the number of anchors that learn an object: in the course's scenes, about 10 per scene. The paper's reason: the easy background anchors add almost nothing to the focal loss, so they should not make it smaller either. Divided by about 2,700, the cross-entropy of the few anchors that learn a shape weighs little next to the box loss, which is a mean over those few: with the mean of the cross-entropy, your FPN detector found only 9 of the 786 shapes, and with the mean of the focal loss, even started at background as below, the course's detector finds only 233 to 286.

        **Start at background.** At the start, before the detector has learned anything, it is about equally unsure of every anchor: all anchors have about the same $p_t$, so the focal loss turns them all down alike, and the background, with 2,697 of the 2,707 anchors of a scene, gives more than 99% of the loss, cross-entropy or focal loss. RetinaNet starts every anchor at background instead: the biases of the head's last layer are set so that every anchor starts with a probability $\pi = 0.01$ of an object. Then the background anchors are easy from the first batch on: on the test scenes, they give 32% of the cross-entropy at the start, and 0.0% of the focal loss. In the paper, the cross-entropy without this start diverged in training; with it, a RetinaNet with a ResNet-50 reached an AP of 30.2 on COCO with the cross-entropy, and 34.0 with the focal loss. Here, the 4 logits of every anchor go through softmax: with a bias $b$ for background and 0 for each of the 3 kinds, softmax gives background $e^b / (e^b + 3)$. Exercise 3 works out $b$.

        **RetinaNet.** The paper's detector is a ResNet with a feature pyramid of five levels, P3 to P7, of 256 channels, 9 anchors on every square, 3 sizes in 3 ratios of width to height, and two heads that all levels share, one for the classes and one for the offsets, each four convolutions of 3 by 3 with ReLU before its last layer. Every one of the 80 classes of COCO gets a sigmoid of its own instead of one softmax, and the loss also weights the anchors of objects by $\alpha = 0.25$ and background by 0.75. With the focal loss, RetinaNet reached an AP of 39.1 on COCO, where the best two-stage detector had 36.8; the paper also trained it on only the hardest anchors, and the best way of doing so reached 32.8 where the focal loss reached 36.0, in a smaller setting with photos of 600 pixels. A footnote of the paper says the focal loss extends easily to more classes at once, and works well there too. Here it is the FPN detector of your last lesson, its one head and its softmax over background and 3 kinds, with $\gamma = 2$, $\pi = 0.01$ and no $\alpha$.
        """),
        L.watch_out("The focal loss is summed over every anchor that is not ignored, background too: that is how the detector learns background. Only the number it is divided by counts the anchors that learn a shape. Summed over the shapes only, the detector never learns what background is, and gives boxes everywhere."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **focal loss** | the cross-entropy of every example times $(1 - p_t)^\gamma$: the surer the model already is of an example, the less it counts | `focal_loss` |
        | **RetinaNet** | a one-stage detector (2017) with a feature pyramid and the focal loss, the first that found objects as well as the two-stage detectors | |
        | $p_t$ | the probability the model gives what an example is: here, the class an anchor learns | `torch.exp(-ce)` |
        | $\gamma$ (gamma) | how fast the weight of an example falls as the model gets surer of it; 0 gives the cross-entropy | `gamma` |
        | $\pi$ (pi) | the probability of an object with which every anchor starts | `pi` |
        | $\alpha$ (alpha) | in the paper, the weight of the anchors of objects (0.25) against background (0.75); not used here | |
        | `F.cross_entropy(..., reduction="none")` | the cross-entropy of every row, a tensor of one number per row, instead of their mean | |
        | `math.log(x)` | the natural logarithm of x | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_focal():
    mo.vstack([
        mo.Html(C.focal_picture({
            "title": "Loss of one anchor", "x": "p_t", "y": "loss", "point": ".",
            "key": ("γ = 0, cross-entropy", "γ = 0.5", "γ = 1", "γ = 2", "γ = 5"),
            "alt": "The loss of one anchor over its p_t, from 0 to 1, for gamma 0, the cross-entropy, as a dashed grey line, and for gamma 0.5, 1, 2 and 5 in purple. All curves start high at a small p_t; the larger gamma, the lower the curve and the sooner it is close to 0. At gamma 2, it is close to 0 from p_t 0.6 on.",
        })),
        mo.md(r"""
        The loss of one anchor over its $p_t$, as in the first figure of the paper. The dashed grey line is the cross-entropy, $\gamma = 0$: still 0.105 at $p_t = 0.9$. The larger $\gamma$, the sooner the loss is close to 0: with $\gamma = 2$, the thick line, from about $p_t = 0.6$ on. For an anchor the detector gets wrong, at a small $p_t$, every curve stays high.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    What does $\gamma$ do to a detector? The slider picks $\gamma$, and with it a detector the course trained with that focal loss, started at background, seed 0. On the left, the loss of one anchor over its $p_t$, with that $\gamma$ thick; on the right, the detector's boxes of 0.5 or more on the scene of the worked example of your last lesson.
    """)
    return


@app.cell(hide_code=True)
def explore_gamma_cell():
    def _frame(g):
        g = float(g)
        _html, _stop = C.gamma_frame(g, {"title": ("γ = {g}: {n} box", "γ = {g}: {n} boxes"), "box": "{kind} {score}", "kinds": ("circle", "square", "triangle"), "point": ".",
                                         "chart": {"title": "Loss of one anchor", "x": "p_t", "y": "loss",
                                                   "alt": "The loss of one anchor over its p_t for gamma 0, 1, 2 and 5, with gamma {g} as a thick purple line and the others thin and grey."},
                                         "alt": "Next to the curves, the scene of the worked example with its five true boxes and the {n} dashed boxes of 0.5 or more of the detector trained with gamma {g}."})
        _weight = "all of its cross-entropy" if g == 0 else f"1/{C.whole(_stop['factor'], L.lang)} of its cross-entropy"
        return _html, f"γ = {g:g}: an anchor at p_t = 0.9 counts {_weight}. Trained with it, the detector finds {_stop['found']} of the 786 shapes of the test scenes, with {_stop['false']} false boxes, and an mAP of {L.number(_stop['map'], 3)}."

    explore_gamma = L.explore("gamma-slider", "γ", _frame, start=2, values=[0, 1, 2, 5], role="",
                              task="Which γ gives the fewest false boxes, and which finds the fewest shapes? What happens to the loss of an anchor at p_t = 0.6 as γ grows?")
    explore_gamma
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Over the seeds 0 to 4, the detectors find about as many shapes with $\gamma$ = 0, 1 and 2: 729 to 740, 726 to 741 and 723 to 734. What changes are the false boxes, the boxes of 0.5 or more that find no shape: 127 to 145 with the cross-entropy, 100 to 127 with $\gamma = 1$, and 95 to 109 with $\gamma = 2$. With $\gamma = 5$, the detector gives the fewest, 71 to 81, but finds fewer shapes, 706 to 711: exercise 5 asks why. In the paper, on COCO, $\gamma = 2$ was best; on these scenes, $\gamma = 1$ does about as well. The table puts the losses side by side, all on the FPN detector of your last lesson, over the seeds 0 to 4:
        """),
        mo.md(C.comparison_table({"header": ("Loss", "Shapes found", "mAP", "mAP 0.75", "False boxes"),
                                  "names": {"mean": "mean of the cross-entropy", "hard": "hard negative mining", "gamma0": "cross-entropy, divided by the shapes",
                                            "focal": "focal loss, γ = 2", "no-start": "focal loss, without the start"}}, L.lang)),
        mo.md(r"""
        Every cell gives the lowest and the highest number over the seeds 0 to 4: how many of the 786 shapes of the 200 test scenes it finds, with a box of their kind, a score of 0.5 or more and an IoU above 0.5; its mAP at an IoU of 0.5 and of 0.75; and its false boxes. The cross-entropy divided by the shapes and the focal loss with γ = 2 start at background; the other rows do not. The focal loss finds as many shapes as hard negative mining, with a few fewer false boxes, and boxes that fit a little less well at an IoU of 0.75. Divided by the shapes, the cross-entropy works too, with more false boxes; the mean of the cross-entropy finds almost nothing, 8 to 59 shapes, and started at background about half of them, 361 to 428. Without the start, the focal loss does about as well here, and so does the cross-entropy divided by the shapes, with 726 to 737 shapes; in the paper, with about 100,000 anchors per photo instead of 2,720, the cross-entropy diverged without it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        A one-stage detector with the focal loss, as RetinaNet:

        1. The FPN detector of your last lesson: the backbone, the feature pyramid, one head, and 2,720 anchors.
        2. For every anchor that is not ignored, its cross-entropy, with `reduction="none"`, and its $p_t = e^{-\text{CE}}$.
        3. Its focal loss: $(1 - p_t)^\gamma$ times its cross-entropy, with $\gamma = 2$.
        4. The sum over those anchors, divided by the number of anchors that learn a shape; plus the box loss.
        5. Start every anchor at background: the bias $b$ of each background logit for which softmax over $b$ and a 0 for every kind gives background $1 - \pi$, and the biases of the kinds 0.
        6. Train and detect as with hard negative mining, with NMS for each kind.
        """),
        L.watch_out("Divide by the number of anchors that learn a shape, `(classes > 0).sum()`, not by every anchor that is not ignored. A mean over about 2,700 anchors makes the loss of the few that learn a shape so small next to the box loss that the detector finds far fewer shapes, with the focal loss too: about a third of them."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example does steps 2 to 4 of the recipe on six made-up anchors, and then shows on a whole scene why step 5 is needed. Each anchor is a row of 4 logits, background first, as your detector gives them, and `classes` holds what each learns, as your `targets` made it: 0 for background, 1 to 3 for a kind, −1 when it is ignored. In step 5, the anchors of the pyramid and `match_anchors` from `everything_ai.scenes` are those of the worked example of your last lesson; the classes of the scene come from the number of the shape every anchor learns, as in your `targets`. `torch.zeros(len(anchors), 4)` gives every anchor the logits of a detector at the start, which is equally unsure of every class. Run the cell as it is.
    """)
    return


@app.cell
def example_focal():
    import math
    import torch
    import torch.nn.functional as F
    from everything_ai.scenes import anchor_boxes, match_anchors, scene

    # 2. six anchors: each a row of 4 logits (background, circle, square, triangle), and the class it
    #    learns; the cross-entropy of every anchor that is not ignored, and p_t = e^(-CE)
    logits = torch.tensor([
        [6.0, 0.0, 0.0, 0.0],   # background, and the detector is sure of it
        [3.0, 0.0, 0.0, 0.0],   # background, and the detector is fairly sure
        [0.0, 2.0, 0.0, 0.0],   # background that looks like a circle
        [0.0, 0.0, 3.0, 0.0],   # ignored
        [0.0, 5.0, 0.0, 0.0],   # a circle, and the detector is sure of it
        [1.0, 0.0, 1.0, 0.0],   # a square, and the detector is not sure
    ])
    classes = torch.tensor([0, 0, 0, -1, 1, 2])
    kept = classes >= 0
    ce = F.cross_entropy(logits[kept], classes[kept], reduction="none")
    p = torch.exp(-ce)
    print(ce)
    print(p)

    # 3. the focal loss of every anchor, with gamma 2, and how many times smaller it is
    fl = (1 - p) ** 2 * ce
    print(fl)
    print(ce / fl)

    # 4. summed, and divided by the number of anchors that learn a shape: the cross-entropy and the
    #    focal loss
    shapes = (classes > 0).sum()
    print(ce.sum() / shapes, fl.sum() / shapes)

    # 5. why every anchor starts at background: the anchors of the pyramid on a scene, what each learns,
    #    and a detector at the start, with every logit 0
    s = scene(26, spread=True)
    anchors = []
    for k, cells in [(2, 32), (3, 16), (4, 8), (5, 4)]:
        side = 2 ** k
        sizes = [(2 * side, 2 * side), (2 * math.sqrt(2) * side, 2 * math.sqrt(2) * side)]
        anchors.append(torch.tensor(anchor_boxes(cells, side, sizes)).float())
    anchors = torch.cat(anchors)
    label = torch.tensor(match_anchors(anchors, [shape.box for shape in s.shapes]))
    scene_classes = torch.zeros(len(anchors), dtype=torch.int64)
    scene_classes[label == -2] = -1
    for i in range(len(s.shapes)):
        scene_classes[label == i] = s.shapes[i].label + 1
    print((scene_classes == 0).sum(), (scene_classes == -1).sum(), (scene_classes > 0).sum())

    start = torch.zeros(len(anchors), 4)
    kept = scene_classes >= 0
    ce = F.cross_entropy(start[kept], scene_classes[kept], reduction="none")
    fl = (1 - torch.exp(-ce)) ** 2 * ce
    background = scene_classes[kept] == 0
    print(ce[0], fl[0])
    print(ce[background].sum() / ce.sum(), fl[background].sum() / fl.sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The background anchor the detector is sure of has $p_t = 0.9926$ and a cross-entropy of 0.0074; its focal loss is 18,354 times smaller, 0.0000004, which Python writes as `4.0365e-07`. The fairly sure one, at 0.87, counts 59 times less, and the circle the detector is sure of 2,547 times less. The two anchors the detector gets wrong keep much more of theirs: the background that looks like a circle, at $p_t = 0.0963$, keeps 1.912 of its 2.341, and the square, at 0.3655, 0.405 of its 1.006. Divided by the 2 anchors that learn a shape, the cross-entropy gives 1.757 and the focal loss 1.160, of which the background that looks like a circle gives 82%.

    The scene has 2,693 background anchors, 10 that are ignored and 17 that learn a shape. At the start, with every logit 0, every anchor has $p_t = 1/4$: a cross-entropy of $\log 4 = 1.386$, and a focal loss of 0.780. The focal loss turns every anchor down alike, so the background gives 99.4% of both. That is what step 5 is for: started at background, the background anchors are easy from the first batch on.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_easy():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A hundred times smaller
    """)
    return


@app.cell(hide_code=True)
def choice_easy_cell():
    choice_easy = L.choice(
        "easy",
        question="With γ = 2, the focal loss of an anchor is (1 − p_t)² times its cross-entropy. The detector gives a background anchor a probability of 0.9 for background. How many times smaller is its focal loss than its cross-entropy?",
        options={
            "right": "100 times: (1 − 0.9)² = 0.01",
            "ten": "10 times: 1 − 0.9 = 0.1",
            "sure": "1.2 times: 0.9² = 0.81",
            "same": "Not smaller: the focal loss only changes the anchors the detector gets wrong",
        },
        correct="right",
        why={
            "right": "(1 − 0.9)² = 0.1² = 0.01: of its cross-entropy, 0.105, a hundredth is left, 0.00105. At p_t = 0.99, a ten-thousandth; at 0.5, a quarter. The surer the detector is, the faster its loss falls.",
            "ten": "That is γ = 1. With γ = 2, the weight 1 − p_t is squared: 0.1² = 0.01, a hundred times smaller.",
            "sure": "0.9² would be a weight of p_t², which makes the anchors the detector is surest of count most. The focal loss weighs by 1 − p_t: (1 − 0.9)² = 0.01.",
            "same": "The focal loss changes every anchor, the easy ones most: (1 − p_t)² is 0.01 at 0.9, and 0.81 at 0.1.",
        },
    )
    choice_easy
    return (choice_easy,)


@app.cell(hide_code=True)
def text_exercise_focal():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The focal loss

        Complete `focal_loss(logits, classes, gamma)`, steps 2 to 4 of the recipe without the box loss: `logits` holds the 4 logits of every anchor of N scenes, a tensor of shape (N, A, 4), background first; `classes` holds what every anchor learns, a tensor of shape (N, A): 0 for background, 1 to 3 for a kind, −1 when it is ignored. It gives back one number, as a tensor: the focal loss of every anchor that is not ignored, summed, and divided by the number of anchors that learn a shape. Your one-stage detector's loss began with `F.cross_entropy(out[:, :, :4][kept], classes[kept])`. The check tries the six anchors of the worked example, two scenes of 40 anchors with five values of gamma, three scenes of 2,720 anchors, and whether training can follow your loss back, as `loss.backward()` does. When it is right, it saves `focal_loss` in your project, as `work/projects/cnn/focal.py`.
        """),
        L.task("Make `focal_loss` give the focal loss of RetinaNet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_focal():
    box_focal = L.code_box("focal-loss", lines=4)
    box_focal
    return (box_focal,)


@app.cell(hide_code=True)
def check_focal(box_focal):
    result_focal = L.check("focal-loss", box_focal.value)
    result_focal
    return (result_focal,)


@app.cell(hide_code=True)
def hints_focal():
    L.hints(
        "focal-loss",
        [
            "Steps 2 to 4 of the worked example, as a function, with `gamma` instead of the 2. There, `kept` picked the anchors that are not ignored, `F.cross_entropy(..., reduction=\"none\")` gave each its cross-entropy, and `torch.exp(-ce)` its p_t.",
            "`kept = classes >= 0` and `ce = F.cross_entropy(logits[kept], classes[kept], reduction=\"none\")`: `logits[kept]` gives the rows of 4 logits of the anchors that are not ignored, one under the other, for any number of scenes. Then `p = torch.exp(-ce)`.",
            "The focal loss of every anchor is `(1 - p) ** gamma * ce`; sum it, and divide by the number of anchors that learn a shape: `return ((1 - p) ** gamma * ce).sum() / (___).sum()`.",
        ],
        why="`classes > 0` is True for every anchor that learns a shape, and `.sum()` counts them: in the six anchors of the worked example 2, the circle and the square. The background anchors are summed in the focal loss too, which is how the detector learns background, but they do not count in what it is divided by.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Start at background

        Complete `background_bias(pi, kinds)`, step 5 of the recipe: `pi` is the probability of a shape with which every anchor should start, and `kinds` the number of kinds besides background. It gives back the bias $b$ of the background logit for which softmax over $b$ and a 0 for every kind gives background a probability of $1 - \pi$. Softmax gives background $e^b / (e^b + \text{kinds})$, since $e^0 = 1$ for every kind: set that equal to $1 - \pi$, and solve it for $b$. The check tries $\pi = 0.01$ with 3 kinds, four other pairs, and softmax over the bias and a 0 for each of 4 kinds. When it is right, it saves `background_bias` in your project, as `work/projects/cnn/background.py`; the demo starts the head of the detector with it.
        """),
        L.task("Make `background_bias` give the bias with which every anchor starts at background."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bias():
    box_background_bias = L.code_box("background-bias", lines=2)
    box_background_bias
    return (box_background_bias,)


@app.cell(hide_code=True)
def check_bias(box_background_bias):
    result_background_bias = L.check("background-bias", box_background_bias.value)
    result_background_bias
    return (result_background_bias,)


@app.cell(hide_code=True)
def hints_bias():
    L.hints(
        "background-bias",
        [
            "Write the equation out: $e^b / (e^b + \\text{kinds}) = 1 - \\pi$. Multiply both sides by $e^b + \\text{kinds}$, and bring the terms with $e^b$ to one side.",
            "$e^b = (1 - \\pi)\\,e^b + (1 - \\pi)\\,\\text{kinds}$, so $\\pi\\,e^b = \\text{kinds}\\,(1 - \\pi)$, and $e^b = \\text{kinds}\\,(1 - \\pi) / \\pi$. With $\\pi = 0.01$ and 3 kinds, that is 297.",
            "$b$ is the logarithm of $e^b$: `return math.log(___)`.",
        ],
        why="With $\\pi = 0.01$ and 3 kinds, $b = \\log 297 \\approx 5.694$. Softmax over (5.694, 0, 0, 0) gives background 297 / (297 + 3) = 0.99, and each kind 1 / 300, about 0.0033: together the 0.01 of $\\pi$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_versus():
    mo.md(r"""
    ### Exercise 4. All or a weight
    """)
    return


@app.cell(hide_code=True)
def choice_versus_cell():
    choice_versus = L.choice(
        "versus",
        question="Hard negative mining and the focal loss both keep the loss of a one-stage detector from drowning in background. A scene has 10 anchors that learn a shape and 2,697 background anchors. How does each of them count the background anchors?",
        options={
            "right": "Hard negative mining counts the 30 hardest fully and the others not at all; the focal loss counts all 2,697, each weighted by how wrong the detector still is",
            "reverse": "Hard negative mining weights every background anchor; the focal loss keeps only the hardest",
            "same": "Both count the 30 hardest background anchors; the focal loss also turns down the anchors of shapes",
            "random": "Hard negative mining picks 30 at random; the focal loss counts all of them equally",
        },
        correct="right",
        why={
            "right": "Hard negative mining makes a cut: an anchor counts fully or not at all, and which 30 change from batch to batch. The focal loss makes no cut: every anchor counts, with a weight of (1 − p_t)^γ, also the anchors of shapes the detector is already sure of. On COCO, the focal loss did better; on the course's scenes, the two find about as many shapes.",
            "reverse": "It is the other way around: hard negative mining keeps the 30 hardest background anchors and drops the others; the focal loss keeps every anchor, weighted by (1 − p_t)^γ.",
            "same": "The focal loss picks no background anchors: it counts all 2,697, the easy ones with a tiny weight. Hard negative mining counts the 30 hardest.",
            "random": "Hard negative mining takes the background anchors with the highest cross-entropy, not random ones; and the focal loss does not count them equally: an easy one counts far less than a hard one.",
        },
    )
    choice_versus
    return (choice_versus,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains the FPN detector of your last lesson with your `focal_loss`, $\gamma = 2$, every anchor started at background with your `background_bias` and $\pi = 0.01$, for 8 epochs on the 2,000 training scenes with small and large shapes, in mini-batches of 32, as it was trained with hard negative mining. It builds the detector with your `pyramid` and `head_rows`, and finds the shapes of the 200 test scenes with your `boxes_from` and NMS. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; that takes a few minutes. While it trains, the chart shows the loss on the training scenes every 10 batches, and dashed, the loss on the 200 test scenes after every epoch. Under the chart are the boxes the detector gives three test scenes at that moment, and under them the feature maps of level 2 of the pyramid, the finest, on the scene of the worked example; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train with the focal loss")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_focal(train_button, result_focal, result_background_bias):
    result_focal, result_background_bias  # run again when an exercise saves your function
    if not train_button.value:
        _out = mo.md("*Press the button to train the detector with your focal loss.*")
    else:
        _now = {}   # the live chart, made at the first report
        _words = {"scene": ("scene {seed}: {n} box", "scene {seed}: {n} boxes"), "layer": "Level 2 of the pyramid on scene {seed}: layer {name}",
                  "alt": "Three test scenes with small and large shapes, each with its true boxes and the dashed boxes of 0.5 or more that the detector gave it after that epoch, after NMS."}
        _words_now = _words | {"alt": "Three test scenes with small and large shapes, each with its true boxes and the dashed boxes of 0.5 or more that the detector gives it now, after NMS."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Focal loss: the loss while it trains", "x": "epoch", "y": "loss",
                                        "alt": "The loss of the detector so far: on the training scenes every 10 batches, and dashed on the 200 test scenes after every epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "training scenes, every 10 batches"), ("loss", True, "test scenes, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-focal-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test scenes after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Focal loss** · epoch {_p['epoch']} of {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:   # the training scenes every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Focal loss** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `focal_loss` and your `background_bias` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than ten minutes. Does your `focal_loss` do more work than it needs, such as a loop over the anchors?*",
                "focal": "*Your `focal_loss` in `work/projects/cnn/focal.py` stops with an error, or does not give one number that training can follow back. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "background": "*Your `background_bias` in `work/projects/cnn/background.py` stops with an error, or does not give a number. Did you change its file? Delete it, and check exercise 3 again.*",
                "pyramid": "*Your `pyramid` in `work/projects/cnn/pyramid.py` stops with an error, or does not give four levels of 32 maps that training can follow back. Did you change its file? Delete it, and check its exercise in [A feature pyramid for small and large objects](/course/open/cv-fpn?lang=en) again.*",
                "head_rows": "*Your `head_rows` in `work/projects/cnn/head_rows.py` stops with an error, or does not give 2,720 rows of 8 for a scene that training can follow back. Did you change its file? Delete it, and check its exercise in [A feature pyramid for small and large objects](/course/open/cv-fpn?lang=en) again.*",
                "offsets": "*Your `boxes_from` in `work/projects/cnn/offsets.py` stops with an error on the detector's offsets. Did you change its file? Delete it, and check its exercise in [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "The loss of the detector after every epoch: on the training scenes solid, on the test scenes dashed."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                _n = sum(len(_b) for _b in _moments[_e - 1])
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"After epoch {_e}: test loss {L.number(_val[_e - 1], 3)}, and {_n} {'box' if _n == 1 else 'boxes'} of 0.5 or more on these three scenes, which have {C.SHOWN_SHAPES} shapes.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: after which epoch does every shape of the three scenes have a box?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _found, _shapes = _result["found"], _result["shapes"]
            _aps = ", ".join(f"{_name} {L.number(_ap, 3)}" for _name, _ap in zip(("circles", "squares", "triangles"), _result["aps"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Started at background with a bias of {L.number(_result['bias'], 3)} and trained with your focal loss, the detector finds {sum(_found)} of the {sum(_shapes)} shapes of the 200 test scenes: a box of their kind, with a score of 0.5 or more and an IoU above 0.5. That is {_found[0]} of the {_shapes[0]} small shapes, {_found[1]} of the {_shapes[1]} medium and {_found[2]} of the {_shapes[2]} large ones. {_result['false']} of its boxes of 0.5 or more find no shape. Its mAP at an IoU of 0.5 is {L.number(_result['map'], 3)}, with an AP of {_aps}; at 0.75, it is {L.number(_result['map75'], 3)}. The course's detector, trained the same way with the seeds 0 to 4, finds 723 to 734 shapes, with 95 to 109 false boxes and an mAP of 0.944 to 0.951; with hard negative mining, it found 727 to 741, with 104 to 114 false boxes.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gamma():
    mo.md(r"""
    ### Exercise 5. Why does a large γ find fewer shapes?
    """)
    return


@app.cell(hide_code=True)
def explain_gamma_cell():
    explain_gamma = L.explain(
        "why-gamma",
        question="With γ = 5, the course's detector gives the fewest false boxes, 71 to 81, but it also finds the fewest shapes, 706 to 711 of the 786, against 723 to 734 with γ = 2. Why does a large γ make it find fewer shapes, and give fewer false boxes? Think of an anchor of a shape whose kind the detector gives a probability of 0.6, and of the score of 0.5 that a box needs.",
        model_answer="With γ = 5, the weight (1 − p_t)⁵ falls very fast: at p_t = 0.6 it is 0.4⁵, about 0.01, so an anchor whose class the detector gives a probability of 0.6 counts only 1% of its cross-entropy, and the loss hardly makes it surer. With γ = 2, it still counts 16%, enough to make it surer. So with γ = 5 the detector stays less sure of the kinds everywhere: on the test scenes, the anchors that learn a shape end at a mean p_t of 0.66, against 0.77 with γ = 2. A box needs a score of 0.5 or more to count, so the shapes whose best anchor stays below it are missed. The same holds for the background: fewer background anchors get a score of 0.5 or more for a kind, so there are fewer false boxes.",
        points=[
            "you say that with γ = 5 the weight (1 − p_t)⁵ is already about 1% at p_t = 0.6, so the loss stops making fairly sure anchors surer",
            "you say that a box needs a score of 0.5, so shapes whose anchors stay around or below it are not found",
            "you say that the detector stays less sure of the kinds everywhere, so fewer background anchors reach a score of 0.5 for a kind either: fewer false boxes",
        ],
    )
    explain_gamma
    return (explain_gamma,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The cross-entropy that your one-stage detector averaged over its anchors in [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) now gets a weight for every anchor, $(1 - p_t)^\gamma$, small for the anchors the detector already gets right, and is divided by the anchors that learn a shape. Like hard negative mining in [A feature pyramid for small and large objects](/course/open/cv-fpn?lang=en), it keeps the few anchors that learn a shape from drowning in background, but without a cut: every anchor counts, as much as the detector is still wrong about it. The two-stage detector of [From R-CNN to Faster R-CNN: two stages](/course/open/cv-rcnn?lang=en) kept the background out with a first stage; RetinaNet does it with a loss, and was the first one-stage detector that found objects as well as the two-stage detectors. Started at background with your `background_bias`, the FPN detector finds as many shapes with your `focal_loss` as with hard negative mining, with a few fewer false boxes. Your project `cnn` now has `focal.py` and `background.py`. In the next lesson, *A kind for every pixel: FCN and transposed convolutions*, a network gives every pixel of a scene a kind, instead of every anchor.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_easy, result_focal, result_background_bias, choice_versus, explain_gamma):
    L.wrap_up(choice_easy, result_focal, result_background_bias, choice_versus, explain_gamma)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
