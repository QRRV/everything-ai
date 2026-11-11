# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-fcn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.
# termen:allow deconvolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-fcn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en) and [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pointwise():
    quick_pointwise = L.choice(
        "quick-pointwise",
        question="A network has a feature map of 64 channels and 16 by 16 places. A 1×1 convolution from 64 to 4 channels: what does it give?",
        options={
            "right": "4 maps of 16 by 16: at each of the 256 places, the same linear layer makes the 64 numbers of that place into 4",
            "whole": "4 numbers for the whole map, as a linear layer at the end of a classifier",
            "neighbors": "4 maps of 16 by 16, in which every place is mixed with its 8 neighbors",
            "smaller": "4 maps of 8 by 8: like pooling, it makes the map half as large",
        },
        correct="right",
        why={
            "right": "A 1×1 convolution looks at one place at a time: it mixes the channels of that place, with the same weights at every place. So the map keeps its 16 by 16 places, and gets 4 channels. Today it gives a score for every class at every place.",
            "whole": "A linear layer at the end would give one answer for the whole map. A 1×1 convolution gives an answer at every place: a linear layer on every pixel. See [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en).",
            "neighbors": "That is a 3×3 convolution. A 1×1 convolution does not look at the neighbors: it only mixes the 64 channels of each place.",
            "smaller": "A 1×1 convolution with stride 1 keeps every place: only the number of channels changes, from 64 to 4.",
        },
    )
    quick_pointwise
    return (quick_pointwise,)


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask-iou",
        question="The mask of a circle has 300 pixels. A model marks 250 pixels as that circle, and 200 of them lie in the mask. What is the IoU of the two masks?",
        options={
            "right": "200 / 350, about 0.57: the union is 300 + 250 − 200 = 350 pixels",
            "found": "200 / 300, about 0.67: the share of the circle that the model found",
            "marked": "200 / 250 = 0.8: the share of the marked pixels that is right",
            "sum": "200 / 550, about 0.36: the overlap divided by both masks together",
        },
        correct="right",
        why={
            "right": "The overlap is 200 pixels, and the two masks cover 300 + 250 − 200 = 350 together: the 200 of the overlap are in both, so they are counted once. With masks, `(a & b).sum()` counts the overlap and `(a | b).sum()` the union.",
            "found": "That counts only the pixels the model missed. A model that marks every pixel of the scene would get 1. The IoU divides by the union: 300 + 250 − 200 = 350. See [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en).",
            "marked": "That counts only the pixels the model marked wrongly. A model that marks a single pixel of the circle would get 1. The IoU divides by the union: 350.",
            "sum": "300 + 250 counts the 200 pixels of the overlap twice. The union counts every pixel once: 350.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A detector gives every shape a box, and a box is rough: its corners are background, and the boxes of two shapes that touch overlap. Cutting a shape out of a photo, or measuring the road ahead of a car or a spot on a scan, needs the class of every pixel: **semantic segmentation**. In 2015, Long, Shelhamer and Darrell showed that a classifier can do it with two changes: no linear layers at the end, so that it takes a picture of any size, and **transposed convolutions**, which make its small feature maps as large as the picture again. Their **FCN**, a fully convolutional network, labeled the pixels of the photos of PASCAL VOC better than every method before it, in less than a fifth of a second per photo, where the best of those took about 50 seconds.
    """)
    return


@app.cell(hide_code=True)
def predict_background():
    guess_background = L.guess(
        "background-share",
        question="A network that says background for every pixel, whatever it sees. On the 200 test scenes of the course, what share of all their pixels does it get right?",
        options={
            "none": "Almost none: it finds no shape",
            "quarter": "About a quarter: background is one of 4 classes",
            "half": "About half",
            "right": "Almost 9 in 10",
        },
        reveal="87%: background covers 87% of the pixels of the test scenes, so a network that has learned nothing gets almost 9 in 10 pixels right. It finds no circle, square or triangle, though: the IoU of the pixels of every kind is 0. That is why this lesson measures every kind on its own.",
    )
    guess_background
    return (guess_background,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A class for every pixel.** In [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), every shape got a mask: `True` for every pixel of that shape. Semantic segmentation gives every pixel of a scene a class: here 0 for background, and 1, 2 and 3 for circle, square and triangle, the label of a shape plus 1. All circles get the same class: it says what a pixel shows, not which shape it belongs to. The **class map**, an array as large as the scene with the class of every pixel, is what the network learns; exercise 1 makes it from the masks.

        **No linear layers.** A classifier, such as your CNNs, ends with linear layers: they want a feature map of one size, and give one answer for the whole picture. FCN treats every linear layer as a convolution whose kernel covers its whole input: a linear layer on a map of 7 by 7 becomes a 7×7 convolution, and the linear layers after it 1×1 convolutions. A 1×1 convolution is a linear layer on every pixel, as in [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en): from the channels of every place of the map, it gives a score for every class at that place. With only convolutions and pooling, a network takes a picture of any size, and its maps grow with it.

        **The course's FCN.** Three blocks of a 3×3 convolution, ReLU and max pooling of 2, with 16, 32 and 64 channels, make a scene of 128 by 128 pixels into a map of 16 by 16: every place of it stands for a square of 8 by 8 pixels. That factor 8 is the **output stride**. One more 3×3 convolution with ReLU, then a 1×1 convolution from 64 channels to 4 gives a score for background and for every kind at every place. With the layer that makes its scores as large as the scene again, below, the network has 64,868 parameters.

        **Upsampling.** The scores must become as large as the scene again, one score per class for every pixel: **upsampling**. In [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en), you made a picture larger in two ways: nearest neighbor repeats every pixel, and bilinear interpolation mixes the old pixels around every new one, the nearer ones more. For the scores of 16 by 16, nearest neighbor gives blocks of 8 by 8 pixels; bilinear interpolation gives soft steps between them.

        **Transposed convolution.** A convolution with stride 2 slides its kernel over the input in steps of 2, and every step gives one number: the map gets smaller. A **transposed convolution** goes the other way: every value of the input paints a copy of the kernel onto the output, times that value, and the next value paints its copy `stride` places further on; where copies overlap, they add up. With H values in a row, a stride $s$ and a kernel of $k$, the output has $(H - 1) \cdot s + k$ values, and padding $p$ cuts $p$ off each end: $(H - 1) \cdot s - 2p + k$. A kernel of ones as large as the stride paints blocks that do not overlap: nearest neighbor. A kernel twice as large as the stride that is largest in the middle and falls off in a straight line to its edges makes the copies of two neighbors overlap and mix: bilinear interpolation, as the worked example shows. And the kernel is a weight like any other, so the network can learn how to upsample, or the kernel can be fixed, for example to bilinear upsampling. The course's FCN starts it as bilinear upsampling, and lets training change it; the table after the slider shows what that gains. In PyTorch, it is `nn.ConvTranspose2d`. The paper calls it a backwards convolution, because it does what a convolution with that stride does in the backward pass; it is also called a deconvolution, though it does not undo a convolution.

        **Training and measuring.** The loss is the cross-entropy of every pixel: `nn.CrossEntropyLoss()` takes scores of shape (N, 4, H, W) and classes of shape (N, H, W), and gives their mean over all pixels. The network's answer for every pixel is the class with the highest score: `out.argmax(1)`. The share of pixels it gets right says little, as the guess showed. So every kind is measured on its own, with the IoU of its pixels, as for masks in [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en): the pixels that are that kind both in the answer and in the class map, divided by the pixels that are that kind in either. This lesson counts every kind over all pixels of the 200 test scenes; the paper also takes the mean over the classes, as your next lesson will.

        **FCN in the paper.** The paper makes the classifier VGG16, trained on the photos of ImageNet, fully convolutional: its last map is 32 times smaller than the photo, and a 1×1 convolution gives 21 scores at every place of it, for the 20 classes of PASCAL VOC and background. One transposed convolution with stride 32, fixed to bilinear upsampling, makes the scores as large as the photo. With the classes of PASCAL VOC 2011, that FCN reached a mean IoU of 56.0% on photos it did not learn from; made in the same way from AlexNet 39.8%, and from GoogLeNet 42.5%. The paper names its limit itself: "The 32 pixel stride at the final prediction layer limits the scale of detail in the upsampled output." So it adds scores from the finer maps of earlier layers, with skip connections, each made twice as large by a transposed convolution that starts as bilinear upsampling and learns: with stride 8, its FCN-8s reached 62.7% where stride 32 reached 59.4%, on part of those photos. Your next lesson builds on that idea.
        """),
        L.watch_out("`nn.CrossEntropyLoss` wants the class map as whole numbers: `dtype=int` in `np.zeros` makes them so, and `torch.tensor` keeps them so. With a class map of floats, it stops with an error."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **semantic segmentation** | giving every pixel of a picture a class: what it shows, not which object it belongs to | `class_map` |
        | **class map** | an array as large as the picture, with the class of every pixel | `classes` |
        | **fully convolutional network** (FCN) | a network of only convolutions and pooling, without linear layers at the end: it takes a picture of any size, and gives a map that grows with it | `FCN` |
        | **output stride** | how many times smaller than the picture the last map of a network is | `stride` |
        | **upsampling** | making a map larger: with nearest neighbor, bilinear interpolation, or a transposed convolution that learns how | |
        | **transposed convolution** | every value of the input paints a copy of the kernel, times that value, `stride` places apart; where copies overlap, they add up | `nn.ConvTranspose2d` |
        | `nn.ConvTranspose2d(c_in, c_out, k, stride=s, padding=p)` | a transposed convolution from `c_in` maps to `c_out` maps; H places become $(H - 1) \cdot s - 2p + k$ | |
        | `nn.CrossEntropyLoss()(out, classes)` | with scores of shape (N, 4, H, W) and classes of shape (N, H, W): the mean cross-entropy of all pixels | |
        | `out.argmax(1)` | for every pixel, the number of the class with the highest score: the place of the largest value along dimension 1, as `np.argmax` with an axis | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        mo.Html(C.steps_picture({
            "scene": "The scene", "coarse": "16 × 16 places", "up": "Upsampled", "truth": "The true classes",
            "kinds": ("circle", "square", "triangle"),
            "alt": "Four pictures of 128 by 128 pixels. The scene of the worked example, with two triangles, a square and a circle. The class with the highest score at every place of the 16 by 16 map of the course's trained FCN: the shapes as rough blocks. The class map after the transposed convolution: the shapes with smooth edges. And the true class map. Every kind has its own color.",
        })),
        mo.md(r"""
        What the course's trained FCN makes of the scene of the worked example. Second, the class with the highest score at every place of its last map, before upsampling: each square stands for 8 by 8 pixels of the scene. Third, the class of every pixel after its transposed convolution, which started as bilinear upsampling and learned: the edges follow the shapes far better than the blocks of 8 by 8 do. Last, the class map it learned from. Here color means a kind, each with its own mark in the key: background is the color of the page.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How coarse may the last map be? The slider picks the output stride, and with it an FCN the course trained, seed 0: with 2, 3, 4 or 5 blocks, each with one more pooling of 2, and the same training. On the left, the scene of the worked example, with one place of the last map around the middle of the circle as a solid square, and the part of the scene that place sees, its receptive field, as a dashed square. On the right, the class of every pixel that FCN gives the scene.
    """)
    return


@app.cell(hide_code=True)
def explore_stride_cell():
    def _frame(s):
        s = int(s)
        _html, _stop = C.stride_frame(s, {"sees": "What one place sees", "title": "Stride {stride}", "kinds": ("circle", "square", "triangle"),
                                          "alt": "Left, the scene of the worked example with one place of the last map, {stride} by {stride} pixels, as a solid square, and the {field} by {field} pixels it sees as a dashed square. Right, the class of every pixel that the FCN with output stride {stride} gives the scene, its last map {places} by {places} places."})
        _ious = ", ".join(f"{_name} {C.span_text(_span, L.lang)}" for _name, _span in zip(("circles", "squares", "triangles"), _stop["ious"]))
        return _html, f"Stride {s}: a last map of {_stop['places']} by {_stop['places']} places, each of which sees {_stop['field']} by {_stop['field']} pixels. The IoU on the test scenes over the seeds 0 to 4: {_ious}."

    explore_stride = L.explore("stride-slider", "stride", _frame, start=8, values=[4, 8, 16, 32], role="",
                               task="Which strides give whole shapes the wrong kind, and which gives one shape several kinds? Compare the dashed square with the shapes, which reach 10 to 22 pixels from their middle.")
    explore_stride
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        The table gives every FCN of the slider over the seeds 0 to 4, with the part of the scene one place of its last map sees. With stride 4, that is only 18 by 18 pixels, less than most shapes, and the network has only 15,492 parameters: it gives parts of one shape different kinds, and its IoU stays at 0.54 to 0.66 for the circles, 0.07 to 0.32 for the squares and 0.30 to 0.46 for the triangles. Stride 8 sees 38 by 38 pixels, about a whole shape, and does best. With strides 16 and 32, a place of the last map stands for 16 by 16 or 32 by 32 pixels, and after 8 epochs the FCN even gives whole shapes the wrong kind. That is not only the coarse map: their networks are one or two blocks deeper, and learn more slowly. Trained twice as long, stride 16 does about as well as stride 8 with four of the five seeds, 0.82 to 0.86 for the circles; with seed 0, it stays at 0.66. Stride 32, trained twice as long, gets to 0.52 to 0.75 for the circles: still well below stride 8.
        """),
        mo.md(C.stride_table({"header": ("Output stride", "Sees", "Circles", "Squares", "Triangles"), "name": "{stride}, {epochs} epochs"}, L.lang)),
        mo.md(r"""
        How much does it matter how the transposed convolution starts, and that it learns? Four FCNs with stride 8, each trained with the seeds 0 to 4 as the course's FCN, the same in everything but their transposed convolution:
        """),
        mo.md(C.upsampling_table({"header": ("Upsampling", "Circles", "Squares", "Triangles", "Pixels right"),
                                  "names": {"s8": "learned, started bilinear", "s8-random": "learned, started at random",
                                            "s8-bilinear": "bilinear, fixed, as in the paper", "s8-nearest": "nearest neighbor, fixed"}}, L.lang)),
        mo.md(r"""
        Started as bilinear upsampling and learned, as in the course's FCN, the transposed convolution does best, with every seed and for every kind. Fixed to bilinear upsampling, as the last one of the paper's FCN is, it does almost as well with seeds 0 and 2, and worse with the others: on these scenes, it helps that training may change the kernel. Started at random, it learns too, but with every seed it ends lower than started as bilinear. Nearest neighbor, fixed, does worst: for every kind, both the lowest and the highest of its five seeds lie below those of every other row. Every edge becomes a staircase of blocks of 8 by 8 pixels. Even a staircase drawn from the true class maps, every block the class that most of its 64 pixels have, would reach only 0.78 for the circles, 0.72 for the squares and 0.64 for the triangles; the FCN with nearest neighbor stays at 0.50 to 0.67, 0.14 to 0.42 and 0.28 to 0.46.

        And an FCN takes a scene of any size. The course's FCN, trained on scenes of 128 pixels, gives every pixel of scenes of 96 and 192 pixels a class without any change, with an IoU for the circles of 0.69 to 0.77 at 96 pixels and 0.82 to 0.92 at 192. At 192 pixels, it does better than at the 128 it learned from. The shapes keep their size, so on a larger scene they overlap less, which probably helps: of the shapes of the 200 test scenes, 12% is partly hidden behind another at 192 pixels, against 25% at 128 pixels and 44% at 96.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        Semantic segmentation with an FCN:

        1. The class map of every scene: 0 everywhere, and the label of a shape plus 1 on the pixels of its mask.
        2. Blocks of a 3×3 convolution, ReLU and max pooling of 2, until the map is $s$ times smaller than the scene; then one more 3×3 convolution with ReLU.
        3. A 1×1 convolution to a score for every class at every place.
        4. A transposed convolution with stride $s$, a kernel of $2s$ and padding $s/2$, started as bilinear upsampling: the scores as large as the scene.
        5. Train with the cross-entropy of every pixel.
        6. For every pixel, the class with the highest score; measure the IoU of the pixels of every kind.
        """),
        L.watch_out("Give the transposed convolution one kernel for every class, from map c to map c, and 0 for the other pairs. With the bilinear kernel for every pair, every class gets the sum of the scores of all classes, and the upsampled scores no longer say which class is highest."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe on the scene of [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), made again with `scene(18)`. The first cell does step 1: `np.zeros` with `dtype=int` makes an array of whole numbers, as large as the scene's picture, and every shape puts its class on the pixels of its mask. Then it does step 6 for a network that says background everywhere. The second cell paints a transposed convolution by hand, step 4 in one row: the row of the guess of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en), 0, 100 and 200, made twice as long with the bilinear kernel of 4, then with a kernel of ones. The third cell builds steps 2 to 5 with layers that have not learned anything yet, and prints the shape after every step. Run the cells as they are.
    """)
    return


@app.cell
def example_classes():
    import numpy as np
    from everything_ai.scenes import scene

    # 1. the class map: 0 for background, the label of a shape plus 1 on the pixels of its mask
    s = scene(18)
    classes = np.zeros(s.image.shape[:2], dtype=int)
    for shape in s.shapes:
        classes[shape.mask] = shape.label + 1
    print([int((classes == k).sum()) for k in range(4)])

    # 6. a network that says background everywhere: the share of pixels right, and the IoU
    #    of the circle's pixels
    pred = np.zeros_like(classes)
    print((pred == classes).mean())
    p = pred == 1
    t = classes == 1
    print((p & t).sum() / (p | t).sum())
    return (classes, np, s)


@app.cell
def example_row():
    import torch

    # 4. a transposed convolution by hand, in one row: every value paints the kernel, times that
    #    value, 2 places further on each time; where the copies overlap, they add up
    row = torch.tensor([0.0, 100.0, 200.0])
    kernel = torch.tensor([0.25, 0.75, 0.75, 0.25])
    stride = 2
    out = torch.zeros((len(row) - 1) * stride + len(kernel))
    for i in range(len(row)):
        out[i * stride:i * stride + len(kernel)] += row[i] * kernel
        print(out)
    print(out[1:-1])   # padding 1: one place off each end

    # nearest neighbor: a kernel of ones as large as the stride
    ones = torch.tensor([1.0, 1.0])
    near = torch.zeros(len(row) * stride)
    for i in range(len(row)):
        near[i * stride:i * stride + len(ones)] += row[i] * ones
    print(near)
    return (torch,)


@app.cell
def example_fcn(classes, np, s, torch):
    from torch import nn

    # 2. blocks of a 3x3 convolution, ReLU and max pooling of 2: the map 8 times smaller
    torch.manual_seed(0)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    convs = nn.ModuleList([nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)])
    pool = nn.MaxPool2d(2)
    for conv in convs:
        x = pool(torch.relu(conv(x)))
        print(x.shape)

    # 3. one more 3x3 convolution, and a 1x1 convolution to a score for every class at every place
    extra = nn.Conv2d(64, 64, 3, padding=1)
    score = nn.Conv2d(64, 4, 1)
    coarse = score(torch.relu(extra(x)))
    print(coarse.shape)

    # 4. a transposed convolution with stride 8, a kernel of 16 and padding 4: as large as the scene
    up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, bias=False)
    out_scene = up(coarse)
    print(out_scene.shape)

    # 5. the cross-entropy of every pixel, against the class map of the first cell
    loss = nn.CrossEntropyLoss()(out_scene, torch.tensor(classes).unsqueeze(0))
    print(loss)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first cell counts 14,622 pixels of background, 694 of the circle, 653 of the square and 415 of the two triangles: 16,384 together, 128 times 128. A network that says background everywhere gets 0.892 of them right, and the IoU of the circle's pixels is 0.0: of its 694 pixels, it calls none a circle. In the second cell, the 0 paints a copy of zeros; the 100 adds 25, 75, 75 and 25 two places further on; and the 200 adds 50, 150, 150 and 50 two places further still, where the copies add up to 75 + 50 = 125 and 25 + 150 = 175. Without the first and the last place, padding 1, that is 0, 25, 75, 125, 175, 150: the row of bilinear interpolation, 0 25 75 125 175 200, except for its last value. Beyond either end of the row, the transposed convolution mixes in a 0: at the end, where no value after the 200 mixes in, that gives 150 instead of 200; at the start, it does not show, because the first value is 0 itself. The kernel of ones gives 0, 0, 100, 100, 200, 200: nearest neighbor. In the third cell, every block makes the map half as large, and the channels grow from 16 to 64: (1, 64, 16, 16) after three blocks. The 1×1 convolution makes that 4 maps of 16 by 16, and the transposed convolution makes them 128 by 128: $(16 - 1) \cdot 8 - 2 \cdot 4 + 16 = 128$. The loss is 1.3862, about $\log 4 = 1.386$: the network has not learned anything yet, so it gives every class about the same score, as if it guessed one of 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_classes():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The class map

        Complete `class_map(s)`, step 1 of the recipe as a function: `s` is a scene of the course, with `s.image`, its picture of shape (rows, columns, 3), and `s.shapes`, its shapes, each with `shape.mask` and `shape.label`. It gives back an array as large as the scene: 0 for background, and the label of a shape plus 1 on the pixels of its mask. The check tries the scene of the worked example, three more scenes, and scenes of 96 and 192 pixels. When it is right, it saves `class_map` in your project, as `work/projects/cnn/classes.py`; the demo makes the classes it learns from with it.
        """),
        L.task("Make `class_map` give the class of every pixel of a scene."),
    ])
    return


@app.cell(hide_code=True)
def exercise_classes():
    box_class_map = L.code_box("class-map", lines=4)
    box_class_map
    return (box_class_map,)


@app.cell(hide_code=True)
def check_classes(box_class_map):
    result_class_map = L.check("class-map", box_class_map.value)
    result_class_map
    return (result_class_map,)


@app.cell(hide_code=True)
def hints_classes():
    L.hints(
        "class-map",
        [
            "The first cell of the worked example, as a function: there, `classes` started as an array of zeros, and every shape put its class on the pixels of its mask.",
            "`np.zeros(s.image.shape[:2], dtype=int)` makes an array of whole numbers as large as the scene's picture, whatever its size. Then `for shape in s.shapes:`, and `classes[shape.mask] = ...` sets every pixel of that shape at once.",
            "The class of a shape is its label plus 1, so that 0 is left for background: `classes[shape.mask] = ___`, and after the loop `return classes`.",
        ],
        why="`shape.label` is 0 for a circle, 1 for a square and 2 for a triangle; `shape.label + 1` makes them the classes 1, 2 and 3. A pixel that no mask covers keeps the 0 of `np.zeros`: background.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_transposed():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A transposed convolution by hand

        Complete `transposed(x, kernel, stride)`, the second cell of the worked example in two directions: `x` is a tensor of shape (H, W), `kernel` one of shape (k, k). Every value `x[i, j]` paints the kernel times that value onto the output, with its top left at row `i * stride` and column `j * stride`; where copies overlap, they add up. It gives back the output, of shape $((H - 1) \cdot s + k, (W - 1) \cdot s + k)$, without padding. The check tries the bilinear kernel of 4 by 4 on a picture of 2 by 2, a kernel of ones, random numbers with strides below, at and above the size of the kernel, and compares your function with PyTorch's `nn.ConvTranspose2d`.
        """),
        L.task("Make `transposed` paint a transposed convolution, copy by copy."),
    ])
    return


@app.cell(hide_code=True)
def exercise_transposed():
    box_transposed = L.code_box("transposed", lines=5)
    box_transposed
    return (box_transposed,)


@app.cell(hide_code=True)
def check_transposed(box_transposed):
    result_transposed = L.check("transposed", box_transposed.value)
    result_transposed
    return (result_transposed,)


@app.cell(hide_code=True)
def hints_transposed():
    L.hints(
        "transposed",
        [
            "In the worked example, `out` started as zeros of length (H − 1) · stride + k, and value `i` added `row[i] * kernel` to the slice from `i * stride` to `i * stride + len(kernel)`. Do the same for the rows and for the columns.",
            "`k = len(kernel)`, and `out = torch.zeros((len(x) - 1) * stride + k, (len(x[0]) - 1) * stride + k)`. Then a loop over the rows, `for i in range(len(x)):`, with a loop over the columns inside it, `for j in range(len(x[0])):`.",
            "Inside both loops, the copy of value `x[i, j]` goes to the rows from `i * stride` and the columns from `j * stride`, `k` of each: `out[i * stride:i * stride + k, j * stride:j * stride + k] += ___`, and after the loops `return out`.",
        ],
        why="`x[i, j] * kernel` is the copy of the kernel for that value, and `+=` adds it to what the copies of its neighbors already painted there. With the bilinear kernel and stride 2, two neighbors overlap by 2 places in each direction, and there they mix.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_upsampler():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Started as bilinear

        Complete `upsampler(classes, stride)`, step 4 of the recipe: it gives back an `nn.ConvTranspose2d` that makes `classes` maps `stride` times larger, with a kernel of 2 · stride, padding stride // 2 and no bias, `bias=False`. It starts as bilinear upsampling: `nn.init.zeros_(up.weight)` sets all its weights to 0, and then map c in gives map c out with `bilinear_kernel(2 * stride)`, which is given. `up.weight[c, c]` is that kernel; PyTorch only lets you set it inside `with torch.no_grad():`. The check tries the course's FCN, 4 maps of 16 by 16 made 8 times larger, other numbers of classes and other strides, and whether training can change it. When it is right, it saves `upsampler` in your project, as `work/projects/cnn/upsample.py`; the demo builds the course's FCN with it.
        """),
        L.task("Make `upsampler` give the transposed convolution of the course's FCN, started as bilinear upsampling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_upsampler():
    box_upsampler = L.code_box("upsampler", lines=5)
    box_upsampler
    return (box_upsampler,)


@app.cell(hide_code=True)
def check_upsampler(box_upsampler):
    result_upsampler = L.check("upsampler", box_upsampler.value)
    result_upsampler
    return (result_upsampler,)


@app.cell(hide_code=True)
def hints_upsampler():
    L.hints(
        "upsampler",
        [
            "The third cell of the worked example made `nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, bias=False)`: 4 maps in, 4 out, a kernel of 16 = 2 · 8 and padding 4 = 8 // 2. Here, `classes` and `stride` take the place of the 4 and the 8.",
            "`up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, padding=stride // 2, bias=False)`, then `nn.init.zeros_(up.weight)`: now every pair of a map in and a map out has a kernel of zeros.",
            "Then `with torch.no_grad():` and, inside it, `for c in range(classes):` with `up.weight[c, c] = ___`; after that `return up`.",
        ],
        why="`up.weight` has shape `(classes, classes, 2 * stride, 2 * stride)`: a kernel for every pair of a map in and a map out. `up.weight[c, c] = bilinear_kernel(2 * stride)` gives map c in to map c out only, so every class is upsampled on its own, and the other pairs stay 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_iou():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. The IoU of one kind

        Complete `kind_iou(pred, classes, kind)`, the end of step 6: `pred` holds the class the network gives every pixel and `classes` the true class of every pixel, both arrays or tensors of the same shape, of one scene or of many. It gives back the IoU of the pixels of `kind`: the pixels that are that kind in both, divided by the pixels that are that kind in either. The first cell of the worked example did it for the circles of a network that says background everywhere. The check tries the scene of the worked example with every pixel moved 3 pixels to the right, that network, and random classes in tensors. When it is right, it saves `kind_iou` in your project, as `work/projects/cnn/kind_iou.py`; the demo measures its FCN with it.
        """),
        L.task("Make `kind_iou` give the IoU of the pixels of one kind."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iou():
    box_kind_iou = L.code_box("kind-iou", lines=3)
    box_kind_iou
    return (box_kind_iou,)


@app.cell(hide_code=True)
def check_iou(box_kind_iou):
    result_kind_iou = L.check("kind-iou", box_kind_iou.value)
    result_kind_iou
    return (result_kind_iou,)


@app.cell(hide_code=True)
def hints_iou():
    L.hints(
        "kind-iou",
        [
            "The first cell of the worked example measured the circles of a network that says background everywhere: the circles are class 1. For another kind, only that number changes.",
            "`p = pred == kind` is `True` for every pixel the network calls that kind, and `t = classes == kind` for every pixel that is that kind. `&` keeps the pixels that are `True` in both, `|` those that are `True` in either.",
            "`return (p & t).sum() / ___`",
        ],
        why="`(p | t).sum()` counts every pixel that is that kind in the answer, in the class map, or in both, once: the union. Divided by it, the overlap gives 1 for a perfect answer, and 0 when the network never calls a pixel of that kind rightly.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains the course's FCN, with output stride 8 and your `upsampler` as its transposed convolution, on the class maps your `class_map` makes of the 2,000 training scenes, for 8 epochs in mini-batches of 32, as the course trained it. After every epoch, it measures the FCN on the 200 test scenes with your `kind_iou`. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; that takes a few minutes. The chart shows the loss on the test scenes after every epoch. When it is done, you can go back through the epochs, and see the classes the FCN gave three test scenes after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train the FCN")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_fcn(train_button, result_class_map, result_upsampler, result_kind_iou):
    result_class_map, result_upsampler, result_kind_iou  # run again when an exercise saves your function
    if not train_button.value:
        _out = mo.md("*Press the button to train the FCN with your functions.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "FCN: the loss on the test scenes", "x": "epoch", "y": "test loss",
                                        "alt": "The loss of the FCN on the 200 test scenes after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**FCN** · epoch {_p['epoch']} of {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:
                _now["chart"].line(f"**FCN** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 1, 3 and 4 first: then your `class_map`, your `upsampler` and your `kind_iou` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than ten minutes. Does one of your functions do more work than it needs, such as a loop over every pixel?*",
                "classes": "*Your `class_map` in `work/projects/cnn/classes.py` stops with an error, or does not give a class from 0 to 3 for every pixel of a scene. Did you change its file? Delete it, and check exercise 1 again: its check then saves your code again.*",
                "upsample": "*Your `upsampler` in `work/projects/cnn/upsample.py` stops with an error, or does not give a transposed convolution that makes the 4 maps of 16 by 16 as large as the scene and that training can change. Did you change its file? Delete it, and check exercise 3 again.*",
                "kind_iou": "*Your `kind_iou` in `work/projects/cnn/kind_iou.py` stops with an error, or does not give 1 for a class map compared with itself. Did you change its file? Delete it, and check exercise 4 again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "The loss of the FCN after every epoch: on the training scenes solid, on the test scenes dashed."}
            _words = {"scene": "scene {seed}", "kinds": ("circle", "square", "triangle"),
                      "alt": "Three test scenes, each with the class of every pixel that the FCN gave it after that epoch."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.rewind_picture(_moments[_e - 1], _words),
                        f"After epoch {_e}: test loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: after which epoch does every shape of the three scenes get its own kind?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _ious = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("circles", "squares", "triangles"), _result["ious"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Trained with your functions, the FCN gets {L.number(100 * _result['acc'], 1)}% of the pixels of the 200 test scenes right, and the IoU of the pixels of every kind is: {_ious}. The course's FCN, trained the same way with the seeds 0 to 4, gets 96.8% to 98.0% of the pixels right, with an IoU of 0.78 to 0.87 for the circles, 0.53 to 0.71 for the squares and 0.60 to 0.73 for the triangles."""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_stride():
    mo.md(r"""
    ### Exercise 5. Why not stride 4?
    """)
    return


@app.cell(hide_code=True)
def explain_stride_cell():
    explain_stride = L.explain(
        "why-stride",
        question="With stride 4, the last map of the course's FCN has 32 by 32 places, four times as many as with stride 8, so its edges could be finer. Yet its IoU is lower for every kind: for the squares 0.07 to 0.32, against 0.53 to 0.71 with stride 8. Why? Look at the dashed square of the slider.",
        model_answer="Every place of the last map gives the scores of its pixels from the part of the scene it sees. With stride 4, that is 18 by 18 pixels, while a shape reaches 10 to 22 pixels from its middle to its edge or its corners: 92% of the shapes of the test scenes are larger than 18 pixels in both directions. A place inside a shape sees a patch of one color, or a piece of straight edge, which looks the same on a square as on a triangle. It cannot tell the kind, so it guesses, and the places of one shape guess differently: in the slider, the square of the scene gets mostly the kind of a circle. With stride 8, a place sees 38 by 38 pixels, about a whole shape with its corners or its round edge. A finer map only gives finer edges once every place sees enough to know the kind. That is the most likely reason, though not the only difference: the network with stride 4 is also smaller, 15,492 parameters against 64,868, with fewer layers to put together what it sees.",
        points=[
            "With stride 4, one place of the last map sees 18 by 18 pixels; with stride 8, 38 by 38.",
            "Most shapes are larger than 18 by 18 pixels, so with stride 4 one place sees only part of a shape, without its corners or its whole edge, and cannot tell its kind.",
            "A finer map only helps the edges once every place sees enough of a shape to know its kind.",
        ],
    )
    explain_stride
    return (explain_stride,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The masks of [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en) became a class map, the IoU of [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en) now measures the pixels of every kind, and the 1×1 convolution of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en) gives a score for every class at every place. The bilinear interpolation of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en) is now the start of a transposed convolution that learns. Where a detector gave every shape a box, the FCN gives every pixel a kind: with your `class_map`, your `upsampler` and your `kind_iou`, the course's FCN with stride 8 reaches an IoU of 0.78 to 0.87 for the circles. Your project `cnn` now has `classes.py`, `upsample.py` and `kind_iou.py`. In the next lesson, *Project: your own U-Net*, a network goes down to coarse maps and back up, and stacks every map on the way up with the sharp map of the same size from the way down, to draw the edges exactly.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_class_map, result_transposed, result_upsampler, result_kind_iou, explain_stride):
    L.wrap_up(result_class_map, result_transposed, result_upsampler, result_kind_iou, explain_stride)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
