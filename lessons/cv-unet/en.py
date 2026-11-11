# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-unet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-unet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_up():
    quick_up = L.choice(
        "quick-up",
        question="A feature map of 64 channels and 16 by 16 places goes into `nn.ConvTranspose2d(64, 32, 2, stride=2)`. What comes out?",
        options={
            "right": "32 maps of 32 by 32: every value paints a kernel of 2 by 2, 2 places further on each time, and the copies just touch",
            "channels": "64 maps of 32 by 32: a transposed convolution keeps the channels",
            "smaller": "32 maps of 8 by 8: stride 2 makes the map half as large",
            "plus": "32 maps of 34 by 34: 16 · 2 + 2 places",
        },
        correct="right",
        why={
            "right": "With $H$ places, stride $s$ and a kernel of $k$, the output has $(H - 1) \\cdot s + k = 15 \\cdot 2 + 2 = 32$ places. A kernel as large as the stride paints copies that touch without overlapping. And the first two numbers say 64 maps in, 32 out. Today every step of the way up starts with such a layer.",
            "channels": "The first two numbers are the maps in and the maps out: from 64 to 32. Like `nn.Conv2d`, a transposed convolution can give any number of maps.",
            "smaller": "That is what a convolution with stride 2 does. A transposed convolution goes the other way: every value paints a copy of the kernel 2 places further on, so the map gets twice as large. See [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).",
            "plus": "The last of the 16 values paints its copy from place $(16 - 1) \\cdot 2 = 30$ on, and the kernel covers 2 places from there: $(H - 1) \\cdot s + k = 32$ places, not $H \\cdot s + k$.",
        },
    )
    quick_up
    return (quick_up,)


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="In the last lesson, the course's FCN with output stride 4 had a last map of 32 by 32 places, and did worse than the FCN with stride 8, whose last map has 16 by 16. Why?",
        options={
            "right": "Every place of its last map sees only 18 by 18 pixels of the scene: too little of a shape to tell its kind",
            "coarse": "Its last map is too coarse to draw the edges",
            "params": "It has more parameters, and learns the training scenes by heart",
            "upsample": "A transposed convolution cannot make a map 4 times larger",
        },
        correct="right",
        why={
            "right": "With two blocks, every place of the last map sees 18 by 18 pixels, while most shapes are larger. With stride 8, a place sees 38 by 38 pixels, about a whole shape, but its map is coarser. An FCN has to choose between seeing enough and drawing fine edges. Today's network does not have to.",
            "coarse": "Its map is finer than that of stride 8: 32 by 32 places against 16 by 16. Its trouble was what one place sees: 18 by 18 pixels, less than most shapes.",
            "params": "It has fewer: 15,492 against 64,868. Its trouble was what one place of its last map sees: 18 by 18 pixels, less than most shapes.",
            "upsample": "It can: with stride 4, a kernel of 8 and padding 2. Its trouble was what one place of its last map sees: 18 by 18 pixels, less than most shapes. See [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        The FCN of the last lesson had to choose. A coarse last map sees enough of the scene to tell a circle from a square, but has only 16 by 16 places to draw the edges from; a fine one could draw finer edges, but each of its places sees too little to know the kind. In 2015, Olaf Ronneberger, Philipp Fischer and Thomas Brox found a way out, for pictures from a microscope, where every cell and every thin wall between two cells counts. Their **U-Net** goes down to coarse feature maps, which see a lot of the picture, and then back up; on the way up, it stacks every map with the map of the same size from the way down. Those **skip connections** give the way up the details from before every pooling of the way down. Trained on only 30 pictures of 512 by 512 pixels, it beat the best method before it on the pictures from an electron microscope of the ISBI challenge, and with the same network it won two parts of the ISBI cell tracking challenge of 2015 by a large margin. Today you build a U-Net yourself, and train it on the course's scenes.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working", "Your CNN on color photos", "Your own one-stage detector",
                            "Your own U-Net", "Final project: ResNet with Grad-CAM", "Final project: U-Net or detector"],
                           6, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_squares():
    guess_squares = L.guess(
        "unet-squares",
        question="The course's FCN reached an IoU of 0.53 to 0.71 for the pixels of the squares. The course's U-Net has 121,556 parameters, and trains on the same 2,000 scenes for 8 epochs, just as the FCN did. What IoU does it reach for the squares, with seed 0?",
        options={
            "worse": "Less: a deeper network learns more slowly",
            "same": "About the same, 0.5 to 0.7",
            "better": "A little better, 0.7 to 0.9",
            "right": "Above 0.9",
        },
        reveal="0.93 with seed 0, against 0.53 to 0.71 for the FCN, and it gets 99.6% of the pixels right. Over the seeds 0 to 4, the median for the squares is 0.92: the middle one of the five. Not every seed does that well, though: with seed 3, the squares reach only 0.60, as the table after the slider shows.",
    )
    guess_squares
    return (guess_squares,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Down and up.** The course's U-Net has two halves. **The way down** is a CNN as you know them: a **block** of two 3×3 convolutions with 8 channels at 128 by 128 pixels, max pooling of 2, a block with 16 channels at 64 by 64, pooling, a block with 32 channels at 32 by 32, pooling, and at the bottom a block with 64 channels at 16 by 16. Every place of that bottom map sees 68 by 68 pixels of the scene, more than the 38 by 38 of the FCN with stride 8: enough for a whole shape. **The way up** goes back in three steps. Each step makes the map twice as large with half the channels, with a transposed convolution of 2 by 2 with stride 2, as in the quick check, and then a block. After the last step, the map is 128 by 128 again, and a 1×1 convolution from 8 channels to 4 gives a score for every class at every pixel, as in the FCN.

        **Skip connections.** The FCN made its scores larger from the coarse map alone: what the poolings threw away was gone. A U-Net keeps the map of every block of the way down, before its pooling. On the way up, after each transposed convolution, it stacks the kept map of the same size in front of it: `torch.cat([kept[2], y], dim=1)` puts the 32 channels of the kept map of 32 by 32, `kept[2]`, and the 32 channels of the upsampled map, `y`, side by side, 64 channels of 32 by 32. The block after it makes those 64 into 32. The kept map has seen little of the scene, but in detail. The upsampled map has seen a lot, but coarsely: enough to know which kind a shape is. The block learns to combine the two. A link that carries a map past the deeper layers like this is a **skip connection**; the U-Net has three, the dashed arrows in the picture below.

        **Stacked, not added.** In [Residual connections](/course/open/nn-residual?lang=en), a connection also skipped layers, but it added: `x + f(x)`, both with the same channels. A skip connection of the U-Net stacks instead, so the block after it gets both maps whole, and learns itself how to weigh them; the price is twice the channels into that block, as exercise 1 counts. The FCN of the paper did yet another thing: it added the scores of an earlier map to the upsampled scores.

        **Batch normalization in every block.** A block of the course's U-Net is two 3×3 convolutions with padding 1, each followed by batch normalization and ReLU. `nn.BatchNorm2d(c)` does what `nn.BatchNorm1d` did in [Batch normalization](/course/open/nn-batchnorm?lang=en), with a channel in the place of a neuron: it standardizes every channel over all its pixels in all scenes of the mini-batch, then multiplies by a learned $\gamma$ and adds a learned $\beta$, one of each per channel. The U-Net of the paper has no batch normalization, which was new then: its paper had come out a few months earlier. Without it, the course's U-Net learns far worse in 8 epochs, as the table after the slider shows.

        **Mean IoU.** In the last lesson, every kind got its own IoU. One number for the whole network is the **mean IoU**: the IoU of the pixels of every class, background included, added up and divided by the number of classes. Every class counts the same, however few pixels it has. A network that says background everywhere gets 87% of the pixels of the test scenes right, but its mean IoU is only 0.22: 0.87 for background, and 0 for each of the three kinds. The FCN paper measured its networks this way, over the 21 classes of PASCAL VOC.

        **U-Net in the paper.** The paper's U-Net is much larger: it goes down four times, from 64 channels to 1,024 at the bottom, with 23 convolutional layers in all. Its convolutions have no padding, so every convolution loses a pixel on every side, and the maps of the way up come out smaller than the kept ones: before stacking, it cuts the middle out of each kept map, "copy and crop" in its figure. A picture of 572 by 572 pixels gives scores for the middle 388 by 388; a larger picture is done in tiles, and around the edge of the picture, the network sees the picture mirrored. The paper describes each step up as making the map larger followed by a 2×2 convolution that halves the channels; the course uses a transposed convolution of 2 by 2 with stride 2, which also makes the map twice as large with half the channels. With so few training pictures, the paper deforms them at random, as if they were made of rubber, and gives the pixels between two cells that touch a larger weight in the loss, so that the network learns to keep the cells apart. Training took 10 hours on a GPU of that time. The course's U-Net pads every convolution with 1 instead, so that the maps keep their size, and the kept map and the upsampled one have the same size without cutting.
        """),
        L.watch_out("`torch.cat` stacks along the dimension you give it. `dim=1` is the channels: (1, 32, 32, 32) and (1, 32, 32, 32) become (1, 64, 32, 32). Without `dim=1`, it stacks along dimension 0, the scenes of the batch: (2, 32, 32, 32), and the block after it stops, because it gets 32 channels where it was made for 64."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **U-Net** | a network that goes down to coarse feature maps and back up, and stacks every map of the way up with the map of the same size from the way down | `UNet` |
        | **the way down** | the half of a U-Net that makes the maps smaller with pooling, and gives them more channels | `self.down` |
        | **the way up** | the half of a U-Net that makes the maps larger again with transposed convolutions | `self.ups` |
        | **skip connection** | a link that carries a map past the deeper layers: in a U-Net, from the way down to the step of the way up with maps of the same size | `torch.cat([kept[2 - i], self.ups[i](x)], dim=1)` |
        | **mean IoU** | the IoU of the pixels of every class, background included, averaged over the classes | `mean_iou` |
        | `torch.cat([a, b], dim=1)` | a and b stacked along dimension 1, the channels: (N, c, H, W) and (N, d, H, W) become (N, c + d, H, W) | |
        | `nn.BatchNorm2d(c)` | batch normalization of c channels: every channel standardized over all its pixels in the mini-batch, then times $\gamma$ plus $\beta$ | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_unet():
    mo.vstack([
        mo.Html(C.unet_picture({
            "down": "The way down", "up": "The way up", "stack": "stack", "pool": "pool",
            "key": ["Every box is a stack of feature maps,", "the number above it its channels."],
            "alt": "The course's U-Net drawn as a U. On the left, the way down: boxes of 8 channels at 128 by 128 pixels, 16 at 64 by 64 and 32 at 32 by 32, each followed by an arrow down for pooling. At the bottom, a box of 64 channels at 16 by 16. On the right, the way up: at 32, 64 and 128 pixels, a dashed box copied from the way down next to the box that came up from below; each pair goes into a block from 2c channels to c. Dashed arrows carry each map of the way down across to the way up. At the end, a 1×1 convolution to 4 maps.",
        })),
        mo.md(r"""
        The course's U-Net. Each box is a stack of feature maps: the taller it is, the larger its maps, and the wider, the more channels. On the left, the way down, with a pooling of 2 after each of the first three blocks. On the right, the way up: each step makes the map from below twice as large, ×2, stacks the dashed copy of the map of the same size from the way down in front of it, and makes the 2c channels into c with a block. The dashed arrows are the three skip connections.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How much does every skip connection give? The slider picks how many of the three skip connections a U-Net keeps, from the bottom up: with 3, the course's U-Net; with 2, the last step, at 128 by 128, gets no kept map; with 0, the way up has only what came out of the bottom. Each of these U-Nets trained just as the course's U-Net, seed 0. On the left, the class of every pixel it gives the scene of the worked example; on the right, the true class map.
    """)
    return


@app.cell(hide_code=True)
def explore_skips_cell():
    def _frame(k):
        k = int(k)
        _skips = f"{k} skip connection" + ("" if k == 1 else "s")
        _html, _stop = C.skips_frame(k, {"title": _skips, "truth": "The true classes", "kinds": ("circle", "square", "triangle"),
                                         "alt": f"Left, the class of every pixel that the U-Net with {_skips} gives the scene of the worked example. Right, the true class map of that scene."})
        _a, _b, _c = (C._FCN.number(_v, 2, L.lang) for _v in _stop["ious"])
        return _html, (f"{_skips}. Seed 0, the U-Net in the picture, on the test scenes: a mean IoU of {C._FCN.number(_stop['miou'], 2, L.lang)}, "
                       f"and an IoU of {_a} for the circles, {_b} for the squares and {_c} for the triangles. "
                       f"Over the seeds 0 to 4, the median mean IoU is {C._FCN.number(_stop['middle'], 2, L.lang)}.")

    explore_skips = L.explore("skips-slider", "skip connections", _frame, start=3, values=[0, 1, 2, 3], role="",
                              task="At 1 skip connection, what does the U-Net make of the two triangles? And from 0 to 3 skip connections, which kind's IoU grows the most?")
    explore_skips
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        The table gives every network over the seeds 0 to 4: the median mean IoU, the middle one of the five when you sort them, the lowest mean IoU, and the median IoU of every kind. Each skip connection helps: from none to three, the median mean IoU goes from 0.91 to 0.93, 0.94 and 0.95. The squares and the triangles, with their corners and straight edges, gain the most: from 0.87 to 0.92 and from 0.84 to 0.92, while the circles go from 0.95 to 0.96. Without skip connections, the network still tells the kinds apart: its bottom map sees enough. Why the squares and the triangles gain the most is exercise 5.
        """),
        mo.md(C.variants_table({"header": ("Network", "Mean IoU", "Lowest", "Circles", "Squares", "Triangles"),
                                "names": {"unet": "U-Net, 3 skip connections", "skips2": "2 skip connections", "skips1": "1 skip connection",
                                          "skips0": "no skip connections", "plain": "3 skip connections, no batch normalization",
                                          "fcn": "the course's FCN, stride 8"}}, L.lang)),
        mo.md(r"""
        The column *Lowest* shows something else. With seed 3, the course's U-Net gets 97.3% of the pixels right, yet its mean IoU is only 0.65: after 8 epochs, it calls 93% of the pixels of the triangles squares, so the IoU of its triangles is 0.04, and that of its squares 0.60. With 1 skip connection, seed 0, the one in the slider, draws the triangles as squares too: the IoU of its triangles is 0.00. Of the 20 U-Nets with batch normalization in the table, these two mix up kinds; the other 18 tell all three apart. The triangles have the fewest pixels of the three kinds, and straight edges and corners like the squares; these two networks call them squares, and the share of pixels right hardly shows it. That is what mean IoU is for. Trained twice as long, for 16 epochs, the U-Net of seed 3 learns the triangles too: a mean IoU of 0.97.

        With these settings, 8 epochs at the learning rate of the FCN, batch normalization matters even more. Without it, the U-Net with all three skip connections reaches a median mean IoU of only 0.43: four of its five seeds stay below 0.25 for the squares and the triangles after 8 epochs. The course's FCN of the last lesson, trained in the same way, reaches 0.77. That comparison says little about the skip connections: the U-Net is also deeper, has batch normalization, and has almost twice the parameters, 121,556 against 64,868. The slider's comparison does say something: the same U-Net, with and without skip connections.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Semantic segmentation with a U-Net:

    1. The way down: a block of two 3×3 convolutions with padding 1, each followed by batch normalization and ReLU; keep its map; max pooling of 2. Three times, with 8, 16 and 32 channels.
    2. The bottom: one more block, with 64 channels, at 16 by 16.
    3. The way up, three times: a transposed convolution of 2 by 2 with stride 2 that halves the channels; the kept map of the same size stacked in front of it, along the channels; a block that halves the channels again.
    4. A 1×1 convolution to a score for every class at every pixel.
    5. Train with the cross-entropy of every pixel, as the FCN.
    6. For every pixel, the class with the highest score; measure the mean IoU: the IoU of every class, background included, averaged.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe on the scene of [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), made again with `scene(18)`. The first cell does steps 1 and 2 with layers that have not learned anything yet, and prints the shape of every kept map and of the bottom. The second cell does the first step of step 3: up to 32 by 32, and the kept map of that size stacked in front. The third cell does step 6 for two made-up networks: one that says background everywhere, and one that gets every pixel right, except that it calls the square a circle. Step 4 is the 1×1 convolution of the last lesson, and step 5 is what the demo does. Run the cells as they are.
    """)
    return


@app.cell
def example_down():
    import torch
    from torch import nn
    from everything_ai.scenes import scene

    # 1. the way down: a block of two 3x3 convolutions with padding 1, each followed by batch
    #    normalization and ReLU; keep its map, then max pooling of 2
    torch.manual_seed(0)
    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    pool = nn.MaxPool2d(2)
    channels = [3, 8, 16, 32]
    kept = []
    for i in range(3):
        conv1 = nn.Conv2d(channels[i], channels[i + 1], 3, padding=1)
        norm1 = nn.BatchNorm2d(channels[i + 1])
        conv2 = nn.Conv2d(channels[i + 1], channels[i + 1], 3, padding=1)
        norm2 = nn.BatchNorm2d(channels[i + 1])
        x = torch.relu(norm1(conv1(x)))
        x = torch.relu(norm2(conv2(x)))
        kept.append(x)
        print(x.shape)
        x = pool(x)

    # 2. the bottom: one more block, from 32 channels to 64, at 16 by 16
    conv1 = nn.Conv2d(32, 64, 3, padding=1)
    norm1 = nn.BatchNorm2d(64)
    conv2 = nn.Conv2d(64, 64, 3, padding=1)
    norm2 = nn.BatchNorm2d(64)
    bottom = torch.relu(norm2(conv2(torch.relu(norm1(conv1(x))))))
    print(bottom.shape)
    return bottom, kept, nn, s, torch


@app.cell
def example_up(bottom, kept, nn, torch):
    # 3. the first step up: twice as large with half the channels, then the kept map of
    #    the same size stacked in front of it, along the channels
    up = nn.ConvTranspose2d(64, 32, 2, stride=2)
    y = up(bottom)
    print(y.shape)
    stacked = torch.cat([kept[2], y], dim=1)
    print(stacked.shape)
    return


@app.cell
def example_miou(s):
    import numpy as np

    # the class map of the scene, as in the last lesson
    classes = np.zeros((128, 128), dtype=int)
    for shape in s.shapes:
        classes[shape.mask] = shape.label + 1

    # 6. the IoU of the pixels of every class, background first, and their mean
    def show_ious(pred):
        total = 0
        for c in range(4):
            p = pred == c
            t = classes == c
            iou = (p & t).sum() / (p | t).sum()
            print("class", c, round(iou, 4))
            total += iou
        print("mean IoU", round(total / 4, 4))

    # a network that says background everywhere
    show_ious(np.zeros_like(classes))

    # a network that gets every pixel right, but calls the square a circle
    mixed = np.zeros_like(classes)
    for shape in s.shapes:
        mixed[shape.mask] = shape.label + 1
    mixed[mixed == 2] = 1
    print("pixels right", (mixed == classes).mean())
    show_ious(mixed)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the first cell, every block keeps the size of its map, thanks to the padding, and gives it more channels: (1, 8, 128, 128), (1, 16, 64, 64) and (1, 32, 32, 32) are kept, and after each, the pooling halves the map. The bottom block gets the map of 16 by 16 and gives (1, 64, 16, 16). In the second cell, the transposed convolution makes that (1, 32, 32, 32): $(16 - 1) \cdot 2 + 2 = 32$ places, and half the channels. It has the same shape as the last kept map, so `torch.cat` can stack the two along the channels: (1, 64, 32, 32), the kept map's 32 channels first. A block from 64 channels to 32 would come next, as in exercise 3. In the third cell, the network that says background everywhere gets an IoU of 0.8925 for background, and 0 for each kind: a mean IoU of 0.2231, though it gets 89% of the pixels right. The network that calls the square a circle gets 96.0% of the pixels right, and an IoU of 1 for background and the triangles. For the circles, it is 0.5152: of the 694 + 653 pixels it calls circle, only the 694 of the circle are right. For the squares, it is 0: it never says square. Its mean IoU is 0.6288: one kind mixed up costs more than a third of the mean, for only 4% of the pixels.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stack():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The channels of a stack
    """)
    return


@app.cell(hide_code=True)
def choice_stack():
    choice_stack_channels = L.choice(
        "stack-channels",
        question="The second step of the way up of the course's U-Net: the map of 32 by 32 with 32 channels comes from below, and the kept map of 64 by 64 has 16 channels. How many channels does the block after the stack get?",
        options={
            "right": "32: 16 from the transposed convolution and 16 from the kept map",
            "add": "16: the two maps are added channel by channel",
            "half": "48: the 32 channels from below and the 16 of the kept map",
            "double": "64: twice the 32 channels from below",
        },
        correct="right",
        why={
            "right": "The transposed convolution makes the 32 channels from below into 16, at 64 by 64. `torch.cat` stacks the 16 of the kept map in front of them: 32 channels. So that block is `DoubleConv(32, 16)`, and gives 16 channels to the next step.",
            "add": "Adding is what a residual connection does. A skip connection of the U-Net stacks, with `torch.cat(..., dim=1)`: the channels of both maps side by side.",
            "half": "The 32 channels from below first go through the transposed convolution, which makes the map twice as large with half the channels: 16. Then the kept 16 are stacked in front: 32.",
            "double": "The transposed convolution halves the channels: from 32 to 16. With the 16 of the kept map, the block gets 32.",
        },
    )
    choice_stack_channels
    return (choice_stack_channels,)


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The block

        Complete the class `DoubleConv(a, b)`, the block of the recipe. In `__init__`: `self.conv1`, a 3×3 convolution with padding 1 from `a` channels to `b`, and `self.norm1`, its batch normalization of `b` channels; then `self.conv2`, from `b` channels to `b`, and `self.norm2`. In `forward`: each convolution, then its batch normalization, then ReLU, twice. The first cell of the worked example did the same in a loop. The check tries the first block of the course's U-Net on two random pictures of 16 by 16, counts its parameters, and tries another block on maps that are not square. When it is right, it saves `DoubleConv` in your project, as `work/projects/cnn/double_conv.py`; your U-Net builds on it.
        """),
        L.task("Make `DoubleConv` two convolutions, each with batch normalization and ReLU."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_double_conv = L.code_box("double-conv", lines=15)
    box_double_conv
    return (box_double_conv,)


@app.cell(hide_code=True)
def check_block(box_double_conv):
    result_double_conv = L.check("double-conv", box_double_conv.value)
    result_double_conv
    return (result_double_conv,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "double-conv",
        [
            "The first cell of the worked example made a block from four layers, `conv1`, `norm1`, `conv2` and `norm2`, and then twice `x = torch.relu(norm1(conv1(x)))`, the second time with `conv2` and `norm2`. In a class, the layers go in `__init__` as `self.conv1` and so on, and the two lines go in `forward`.",
            "`self.conv1 = nn.Conv2d(a, b, 3, padding=1)` and `self.norm1 = nn.BatchNorm2d(b)`. The second convolution gets the `b` channels of the first: `nn.Conv2d(b, b, 3, padding=1)`, and `self.norm2` is again of `b` channels.",
            "In `forward`: `x = torch.relu(self.norm1(self.conv1(x)))`, and then `return ___`, with the second convolution, its batch normalization and ReLU, as in the line before.",
        ],
        why="A 3×3 convolution makes every channel from all channels of the 3 by 3 pixels around a place; padding 1 adds a border of zeros, so that the map keeps its size. Batch normalization then standardizes every channel over the mini-batch, and ReLU makes the negative values 0. Two convolutions in a row see 5 by 5 pixels of the map they get.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unet():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your U-Net

        Complete the class `UNet`. `__init__` already has the way down, `self.down`, four blocks made with your `DoubleConv`; the pooling, `self.pool`; the way up, `self.ups`, three transposed convolutions; and the 1×1 convolution, `self.score`. Fill in `self.convs`, the three blocks after the skip connections: each gets the stacked channels and gives half of them. Then `forward`. The way down keeps the map of each of the first three blocks in a list `kept`, before its pooling, and ends with the bottom block, without pooling. The way up makes the map twice as large in every step `i` with `self.ups[i]`, stacks the kept map of the same size in front of it along the channels, and gives that to `self.convs[i]`. At the end comes `self.score`. The check tries two random pictures of 32 by 32, counts the parameters, and tries a picture of 48 by 64. When it is right, it saves `UNet` in your project, as `work/projects/cnn/unet.py`; the demo trains it.
        """),
        L.task("Make `UNet` go down and back up, with a skip connection at every size."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unet():
    box_unet = L.code_box("unet", lines=26)
    box_unet
    return (box_unet,)


@app.cell(hide_code=True)
def check_unet(box_unet):
    result_unet = L.check("unet", box_unet.value)
    result_unet
    return (result_unet,)


@app.cell(hide_code=True)
def hints_unet():
    L.hints(
        "unet",
        [
            "`self.convs` is a `nn.ModuleList` of three `DoubleConv`, like `self.down`. At 32 by 32, the kept map has 32 channels and the upsampled one too, as in the second cell of the worked example: how many channels does that block get, and how many does it give? Exercise 1 did the step after it.",
            "The way down: `kept = []`, then `for i in range(3):` with `x = self.down[i](x)`, `kept.append(x)` and `x = self.pool(x)`; after the loop, `x = self.down[3](x)`. The way up: `for i in range(3):`, where step 0 needs the map kept last, at 32 by 32, and step 2 the one kept first, at 128 by 128.",
            "Inside the loop of the way up: `x = self.convs[i](torch.cat([kept[2 - i], ___], dim=1))`, with the map from below made twice as large; after the loop, `return self.score(x)`.",
        ],
        why="The maps are kept from large to small, so the way up, from small to large, takes them in the opposite order: `kept[2]` at 32 by 32 first, `kept[0]` at 128 by 128 last. `torch.cat(..., dim=1)` stacks along the channels, so the block after it gets both maps whole: the kept map first, as in the paper's figure.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_miou():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Mean IoU

        Complete `mean_iou(pred, classes, n)`, step 6 of the recipe: `pred` and `classes` as in `kind_iou`, of one scene or of many, and `n` the number of classes. It gives back the IoU of every class from 0 to n − 1, background included, averaged over the n classes. The file imports your `kind_iou` from the last lesson: use it. The check tries the scene of the worked example with the square called a circle, the network that says background everywhere, and random classes in tensors, with 3 classes. When it is right, it saves `mean_iou` in your project, as `work/projects/cnn/mean_iou.py`; the demo measures your U-Net with it.
        """),
        L.task("Make `mean_iou` give the IoU of every class, averaged."),
    ])
    return


@app.cell(hide_code=True)
def exercise_miou():
    box_mean_iou = L.code_box("mean-iou", lines=8)
    box_mean_iou
    return (box_mean_iou,)


@app.cell(hide_code=True)
def check_miou(box_mean_iou):
    result_mean_iou = L.check("mean-iou", box_mean_iou.value)
    result_mean_iou
    return (result_mean_iou,)


@app.cell(hide_code=True)
def hints_miou():
    L.hints(
        "mean-iou",
        [
            "The third cell of the worked example did it for 4 classes: the IoU of every class added up in `total`, then divided by 4. Here, `kind_iou(pred, classes, c)` gives the IoU of class `c`, and `n` takes the place of the 4.",
            "`total = 0`, then `for c in range(n):` with `total += kind_iou(pred, classes, c)`.",
            "After the loop: `return ___`.",
        ],
        why="`range(n)` starts at 0, so background counts as a class like the kinds. Every class adds its IoU once, whatever its number of pixels: a class of a few pixels weighs as much as background.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains your `UNet`, made of your `DoubleConv`, on the class maps your `class_map` makes of the 2,000 training scenes, for 8 epochs in mini-batches of 32, just as the course's FCN trained. Then it measures your U-Net on the 200 test scenes with your `mean_iou`. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; on the course's computer, with 4 cores, that took about 9 minutes. While it trains, the chart shows the loss on the training scenes every 10 batches, and dashed, the loss on the 200 test scenes after every epoch. Beside it are the classes your U-Net gives three test scenes at that moment, and under them the feature maps of its last block before the scores, on the scene of the worked example; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your U-Net")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_unet(train_button, result_double_conv, result_unet, result_mean_iou):
    result_double_conv, result_unet, result_mean_iou  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your U-Net.*")
    else:
        _now = {}   # the live chart, made at the first report
        _words = {"scene": "scene {seed}", "kinds": ("circle", "square", "triangle"), "layer": "Layer {name} of your U-Net on scene {seed}",
                  "alt": "Three test scenes, each with the class of every pixel that your U-Net gave it after that epoch."}
        _words_now = _words | {"alt": "Three test scenes, each with the class of every pixel that your U-Net gives it now."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your U-Net: the loss while it trains", "x": "epoch", "y": "loss",
                                        "alt": "The loss of your U-Net so far: on the training scenes every 10 batches, and dashed on the 200 test scenes after every epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "training scenes, every 10 batches"), ("loss", True, "test scenes, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-unet-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test scenes after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your U-Net** · epoch {_p['epoch']} of {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:   # the training scenes every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your U-Net** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 3 and 4 first: then your `UNet` and your `mean_iou` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than half an hour. Does your U-Net do more work than it needs, such as more blocks or more channels than the recipe? If it follows the recipe, this computer is too slow for the demo.*",
                "classes": "*Your `class_map` in `work/projects/cnn/classes.py` stops with an error, or does not give a class from 0 to 3 for every pixel of a scene. Did you change its file? Delete it, and check its exercise in [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) again.*",
                "unet": "*Your `UNet` in `work/projects/cnn/unet.py` stops with an error, does not give 4 scores for every pixel of a scene, or trains to a loss that is not a number. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
                "mean_iou": "*Your `mean_iou` in `work/projects/cnn/mean_iou.py` stops with an error, or does not give 1 for class maps compared with themselves. Did you change its file? Delete it, and check exercise 4 again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "The loss of your U-Net after every epoch: on the training scenes solid, on the test scenes dashed."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"After epoch {_e}: test loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: which kind does your U-Net find last?")]
            else:
                _pic = [mo.Html(C._FCN.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _ious = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("background", "circles", "squares", "triangles"), _result["ious"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Your U-Net, with {C._FCN.whole(_result['params'], L.lang)} parameters, gets {L.number(100 * _result['acc'], 1)}% of the pixels of the 200 test scenes right. Its mean IoU, by your `mean_iou`, is {L.number(_result['miou'], 3)}, with an IoU of: {_ious}. With seed 0, the run the demo follows, the course's U-Net gets 99.6% of the pixels right, with a mean IoU of 0.953; over the seeds 0 to 4, the median mean IoU is 0.95, and with seed 3 it is only 0.65, as the table after the slider over the skip connections shows.{_who}"""),
                mo.md("**Your U-Net, layer by layer.** The first box is the scene of the worked example, and the boxes after it are the layers of your trained U-Net, in the order the scene goes through them; under them are the feature maps of the layer you choose."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Go through your trained U-Net from the scene to the scores: what does each layer make of the scene?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
                mo.md("On the way down, the maps get coarser: in the first block, many follow the edges of the shapes or the shapes themselves, and at the bottom, on only 16 by 16 places, they are coarse spots. On the way up, the map that comes out of `ups.2` is still blurred; stacked with the map kept on the way down, the edges are sharp again in `convs.2`. The last layer, `score`, has one channel for every class: the circle has the most ink in channel 1, the square in channel 2 and both triangles in channel 3; in channel 0, the background, it is the other way round: there the shapes have the least ink."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_skips():
    mo.md(r"""
    ### Exercise 5. What the skip connections give
    """)
    return


@app.cell(hide_code=True)
def explain_skips_cell():
    explain_skips = L.explain(
        "why-skips",
        question="Without skip connections, the course's U-Net reaches a median mean IoU of 0.91 over the seeds 0 to 4; with all three, 0.95. The way down is the same in both. What do the skip connections give the way up that the bottom map cannot? Look at the squares and the triangles in the table.",
        model_answer="The bottom map has 16 by 16 places. Each of them sees 68 by 68 pixels of the scene, enough to know which kind a shape is, but stands for 8 by 8 pixels, and after three poolings, each of which kept only the largest of 4 values, it says less exactly where in those pixels an edge or a corner lies. Without skip connections, the way up has to draw every shape from that coarse map. With them, every step up also gets the map of the same size from the way down, from before its pooling: it has seen only a small part of the scene, but in detail. The block after the stack combines the two: the kind from the map that came up from below, the details from the kept map. The edge of a circle follows from where its middle is and how large it is, which the coarse map knows; a corner is a small detail that a place of 8 by 8 pixels blurs. That is most likely why the squares and the triangles gain the most, from 0.87 to 0.92 and from 0.84 to 0.92, while the circles, which the network without skip connections already draws well, go only from 0.95 to 0.96.",
        points=[
            "The bottom map is coarse: each of its places stands for 8 by 8 pixels, and after three poolings it says less exactly where an edge or a corner lies.",
            "A skip connection brings the map of the same size from the way down, from before its pooling: it has seen little of the scene, but in detail.",
            "The block after the stack combines the two: the kind from the map that came up from below, the details from the kept map.",
        ],
    )
    explain_skips
    return (explain_skips,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The transposed convolution of [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) now makes the maps of a U-Net twice as large in every step of the way up, the batch normalization of [Batch normalization](/course/open/nn-batchnorm?lang=en) now works on every channel of a feature map, and where a connection in [Residual connections](/course/open/nn-residual?lang=en) added a map, a skip connection of the U-Net stacks it. The IoU of every class, background included, has become one number, the mean IoU. Your `UNet`, built from your `DoubleConv`, is the course's U-Net: over five seeds, it reaches a median mean IoU of 0.95 on the course's scenes, and the course's FCN, measured in the same way, 0.77. Your project `cnn` now has `double_conv.py`, `unet.py` and `mean_iou.py`. In the next lesson, *Dilated convolutions and DeepLab*, a convolution spreads its kernel out with gaps: it sees a larger part of the scene without making the map smaller.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stack_channels, result_double_conv, result_unet, result_mean_iou, explain_skips):
    L.wrap_up(choice_stack_channels, result_double_conv, result_unet, result_mean_iou, explain_skips)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
