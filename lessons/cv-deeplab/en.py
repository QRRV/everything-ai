# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-deeplab. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-deeplab", "en")
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
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="The course's FCN with output stride 16 has four blocks, each with a pooling of 2. For a scene of 128 by 128 pixels, how large is its last map, and how many pixels of the scene does one place of it stand for?",
        options={
            "right": "8 by 8 places, each for 16 by 16 pixels",
            "eight": "16 by 16 places, each for 8 by 8 pixels",
            "four": "32 by 32 places, each for 4 by 4 pixels",
            "same": "128 by 128 places: padding keeps the map as large as the scene",
        },
        correct="right",
        why={
            "right": "Every pooling halves the map: 128, 64, 32, 16, 8. So 8 by 8 places are left, and each stands for 128 / 8 = 16 by 16 pixels: an output stride of 16. Its class maps were coarse, and after 8 epochs its IoU stayed below that of stride 8. Today you take out the last two of those poolings, and the map stays 32 by 32.",
            "eight": "That is the FCN with stride 8, with three poolings. A fourth halves the map once more: 8 by 8 places, each for 16 by 16 pixels.",
            "four": "That is a map after two poolings. Four poolings halve the map four times: 128 / 16 = 8, so 8 by 8 places.",
            "same": "Padding keeps the map as large as it was through a convolution, but every pooling of 2 halves it: four poolings make 128 into 8.",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def quick_check_up():
    quick_up = L.choice(
        "quick-up",
        question="The upsampler of the last lesson with stride 4 is `nn.ConvTranspose2d(4, 4, 8, stride=4, padding=2)`. It gets 4 maps of 32 by 32. How large are the maps it gives?",
        options={
            "right": "128 by 128: (32 − 1) · 4 − 2 · 2 + 8",
            "no-padding": "132 by 132: (32 − 1) · 4 + 8",
            "times": "136 by 136: 32 · 4 + 8",
            "double": "64 by 64: a transposed convolution makes a map twice as large",
        },
        correct="right",
        why={
            "right": "With $H$ places, stride $s$, padding $p$ and a kernel of $k$, the output has $(H - 1) \\cdot s - 2p + k = 31 \\cdot 4 - 4 + 8 = 128$ places: 4 times as large. Today's network ends with this upsampler, on a map of 32 by 32.",
            "no-padding": "Padding 2 cuts 2 places off each end: $(H - 1) \\cdot s - 2p + k = 124 - 4 + 8 = 128$.",
            "times": "The last of the 32 values paints its copy from place $(32 - 1) \\cdot 4 = 124$ on, not from $32 \\cdot 4$; and padding 2 cuts 2 places off each end: 128.",
            "double": "Only with stride 2. With stride 4, every value paints its copy 4 places further on: about 4 times as large, $(32 - 1) \\cdot 4 - 2 \\cdot 2 + 8 = 128$. See [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en).",
        },
    )
    quick_up
    return (quick_up,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A CNN that recognizes photos pools again and again: its last map is 32 times smaller than the photo, and every place of it sees a large part of the photo. An FCN made of such a network makes coarse class maps, as you saw in the last lesson with stride 16. Fewer poolings would give a finer map, but then the layers after them see less of the photo than they learned to look at. In 2015, Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy and Alan Yuille found a way out for a VGG-16 trained on ImageNet: they took out its last two poolings, and spread the kernels of the layers after them out with gaps. Every layer still saw what it had learned to see, and the last map became 8 times smaller than the photo instead of 32. Their **DeepLab** later added several of those spread-out convolutions side by side, to see small and large surroundings at once, and reached a mean IoU of 79.7% on the test photos of PASCAL VOC 2012, where the FCN-8s of the last lesson had 62.2%. Today you build a small DeepLab from the layers of the course's FCN, and train it on the course's scenes.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_field():
    guess_field = L.guess(
        "dilated-field",
        question="A 3×3 convolution has its 9 taps next to each other, and sees 3 by 3 pixels. Spread its taps out so that they are 2 pixels apart, with one pixel between them, and it sees 5 by 5 pixels, with the same 9 weights. Stack three 3×3 convolutions, the first with its taps 1 pixel apart, the second 2 apart and the third 4 apart. How wide is the part of the picture one place after the third sees?",
        options={
            "seven": "7 pixels: each 3×3 convolution adds 2",
            "nine": "9 pixels: the widest of the three",
            "right": "15 pixels",
            "twentyseven": "27 pixels: 3 · 3 · 3",
        },
        reveal="15: the first adds 2 to the 1 pixel of a place, the second 2 · 2 = 4, the third 2 · 4 = 8, and 1 + 2 + 4 + 8 = 15. Three plain 3×3 convolutions see only 7. The second cell of the worked example measures it, and exercise 1 counts it for any stack.",
    )
    guess_field
    return (guess_field,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Gaps between the taps.** A 3×3 convolution multiplies 9 pixels next to each other, its **taps**, by its 9 weights, and adds them up. A **dilated convolution** spreads those taps out: with **dilation** 2, they are 2 pixels apart, and it sees 5 by 5 pixels; with dilation 4, they are 4 apart, and it sees 9 by 9. It still has 9 weights, and it skips the pixels between its taps. A kernel of $k$ with dilation $d$ spans $k + (k - 1)(d - 1)$ pixels, for a 3×3 kernel $2d + 1$. It does the same as a 5×5 kernel with zeros between its taps, as the first cell of the worked example shows, but multiplies only the 9 taps. In PyTorch, `nn.Conv2d(c_in, c, 3, padding=d, dilation=d)`: padding $d$ keeps the map as large as it was. The DeepLab papers call it an atrous convolution, from the French *à trous*, with holes. <!-- termen:ok -->

        **Stacked.** A layer with dilation $d$ makes the part of the picture one place sees $(k - 1) \cdot d$ wider, times the step between neighbors of its input, as in [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en). Three 3×3 convolutions with dilation 1, 2 and 4 see 15 pixels, as in the guess; with one more of dilation 8, 31. With every dilation twice the one before, what one place sees about doubles with every layer, while every layer adds the same number of weights. A pooling of 2 also lets the layers after it see twice as far, but it halves the map.

        **A pooling taken out.** That is what DeepLab does with a CNN that already learned. It sets the stride of a pooling to 1: the pooling still takes the largest of every 2 by 2 values, but moves 1 place at a time, so the map stays twice as large. Every layer after it gets dilation 2. The layer that came after the pooling now gets a map whose neighbors are twice as close; with its taps 2 places apart, it reaches exactly the values it reached before. At every other place of the larger map, it gives the same numbers as before, as the third cell of the worked example shows, and in between it gives new ones. The layers after a second pooling taken out get dilation 4. The trained weights keep working, and the class map comes out finer.

        **Atrous spatial pyramid pooling.** Some shapes need a small surrounding to tell their kind, others a large one. **ASPP**, atrous spatial pyramid pooling, puts several convolutions side by side on the same map: a 1×1 convolution, and a 3×3 convolution for every dilation, the course's with dilation 2, 4 and 8. Each branch gives `c` channels, followed by ReLU. `torch.cat(maps, dim=1)` stacks their maps along the channels, and a last 1×1 convolution with ReLU makes those 4 · `c` channels into `c` again: it learns how to combine what each branch saw. The pyramid is the several sizes of surroundings, like a picture at several scales; despite the name, the course's ASPP pools nothing.

        **The course's DeepLab.** It has the layers of the course's FCN with stride 16: 3×3 convolutions with 16, 32, 64 and 64 channels. It learns from scratch, so it has no trained weights to keep, and leaves the third and the fourth pooling out altogether: only the first two convolutions are followed by a pooling of 2, and its last map is 32 by 32, 4 times smaller than the scene. conv4 came after the third pooling, so it gets dilation 2; the end of the FCN, one more 3×3 convolution, came after the fourth, so it would get dilation 4: the course puts an ASPP there, with its dilations 2, 4 and 8 around that 4. Then a 1×1 convolution to a score for every class, and your upsampler of the last lesson with stride 4. Exercise 3 builds it.

        **DeepLab in the papers.** The first DeepLab, of 2015, starts from a VGG-16 trained on ImageNet. It sets the stride of the last two of its five poolings to 1 and dilates the layers after them by 2 and 4, so that its last map is 8 times smaller than the photo instead of 32. It makes its scores 8 times larger with bilinear interpolation, which learns nothing, and a separate step after the network then makes the edges of the classes follow the edges in the photo. DeepLab v2, of 2016, adds ASPP: four branches side by side, each starting with a 3×3 convolution with dilation 6, 12, 18 or 24, whose scores are added up. With a ResNet-101 and a few more steps, it reached the 79.7% of *Why this matters*. DeepLab v3, of 2017, has the ASPP the course's is modeled on: a 1×1 convolution and three 3×3 convolutions with dilation 6, 12 and 18, each with 256 channels and batch normalization, stacked. Its authors noticed that with a dilation almost as large as the map, a 3×3 convolution reaches mostly padding outside its middle tap, and works like a 1×1 convolution; so they added a branch that averages the whole map, for what the photo as a whole shows. Without the separate step at the end, v3 reached 85.7%. The course's ASPP has neither batch normalization nor that branch.
        """),
        mo.Html(C.taps_picture({"title": "dilation {d}", "sees": "sees {n} × {n}",
                                "alt": "Three grids of 9 by 9 pixels, each with the 9 taps of a 3×3 kernel colored: with dilation 1 next to each other in the middle, with dilation 2 two pixels apart, spanning 5 by 5, and with dilation 4 four pixels apart, spanning 9 by 9. A dashed square marks the part each sees."})),
        mo.md(r"""
        The 9 taps of a 3×3 kernel with dilation 1, 2 and 4: the same 9 weights, spread over 3 by 3, 5 by 5 and 9 by 9 pixels. The dashed square is the part of the picture one place sees; the pixels between the taps, it skips.
        """),
        L.watch_out("In [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en), dilation made the white shapes of a binary image grow. The dilation of a convolution is something else: how far apart its taps are. With `dilation=1`, the default, they are next to each other."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **dilated convolution** | a convolution whose taps are spread out with gaps between them: it sees a larger part of the map with the same weights | `nn.Conv2d(c_in, c, 3, padding=d, dilation=d)` |
        | **dilation** | how many places apart the taps of a convolution are; 1 is next to each other | `dilation=d` |
        | **tap** | one place of the input that a kernel multiplies by one of its weights | |
        | **ASPP** | atrous spatial pyramid pooling: a 1×1 convolution and 3×3 convolutions with several dilations side by side on one map, their maps stacked and combined by a 1×1 convolution | `ASPP` |
        | **DeepLab** | a network for semantic segmentation that takes out the last poolings of a CNN, dilates the layers after them, and ends with an ASPP | `DeepLab` |
        | `nn.MaxPool2d(2, stride=1)` | max pooling of 2 by 2 that moves 1 place at a time: the map loses only one place | |
        | `x[:, :, ::2, ::2]` | every second place of the maps, along both sides, from the first: `::2` is a slice with step 2, like `range(0, n, 2)` | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_maps():
    mo.vstack([
        mo.Html(C.maps_picture({"scene": "The scene", "fcn": "FCN, stride 16", "deeplab": "DeepLab", "truth": "The true classes",
                                "kinds": ("circle", "square", "triangle"),
                                "alt": "Four pictures side by side: the scene of the worked example; the class of every pixel that the course's FCN with stride 16 gives it, coarse blobs; the class of every pixel that the course's DeepLab gives it, close to the true shapes; and the true class map."})),
        mo.md(r"""
        The scene of the worked example, and the classes that two networks with the same four convolutions give it, each trained just as the FCN of the last lesson, seed 0. The FCN with stride 16 pools four times, and draws its classes from a map of 8 by 8. The course's DeepLab pools twice, dilates conv4 and its ASPP, and draws from a map of 32 by 32.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Spread the taps too far, and the layers skip pixels. The slider picks the dilation of a second 3×3 convolution after a first one with dilation 1. The grid is the picture around one place, the framed square in the middle: a pixel the place sees is colored, and the more paths lead from it through the two layers to the place, the fuller its color; a pixel it skips stays empty. The dashed square is the part it spans.
    """)
    return


@app.cell(hide_code=True)
def explore_gaps_cell():
    def _frame(d):
        d = int(d)
        _html, _stop = C.gaps_frame(d, {"title": f"3×3, then 3×3 with dilation {d}", "key": "the fuller, the more paths",
                                        "alt": "A grid of 15 by 15 pixels around one place, with the pixels that a 3×3 convolution with dilation 1 followed by one with dilation {d} sees colored, and the pixels it skips empty."})
        _skips = "it sees every pixel in it" if _stop["gaps"] == 0 else f"it sees {_stop['seen']} of them and skips {_stop['gaps']} in between"
        return _html, f"Dilation {d}: the place spans {_stop['width']} by {_stop['width']} pixels, {_stop['width'] ** 2} in all, and {_skips}."

    explore_gaps = L.explore("gaps-slider", "dilation", _frame, start=2, values=list(C.GAPS), role="",
                             task="From which dilation on does the place skip pixels? And how many pixels does it see from there on?")
    explore_gaps
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Up to dilation 3, the place sees every pixel it spans: around every tap of the second layer, the first covers 3 pixels in a row, so taps up to 3 apart leave no gap. From dilation 4 on, the taps are further apart than that, and the place sees 81 pixels, the 3 by 3 of the first layer around each of the 9 taps of the second, however far apart: at dilation 4 it spans 11 by 11 pixels and skips 40, at 6 it spans 15 by 15 and skips 144. That is why the dilations of a stack grow step by step: 1, 2, 4 and 8 leave no gap, as the second cell of the worked example shows. In the course's DeepLab, conv3, conv4 and the branch with dilation 8 are a stack of 1, 2 and 8: it skips some places of the map conv3 gets. The branches with dilation 2 and 4 see every place they span.

        The table gives the course's networks over the seeds 0 to 4, each trained just as the FCN of the last lesson: the side of the last map before the upsampler, the pixels one place of it sees, for an ASPP from its 1×1 branch to its branch with the largest dilation, and the IoU of the pixels of every kind on the test scenes, from the lowest to the highest over the seeds. *DeepLab, ASPP 1, 1, 1* is the course's DeepLab with dilation 1 in every branch of its ASPP; *dilated, no ASPP* has conv4 dilated by 2 and, in place of the ASPP, the last 3×3 convolution of the FCN, dilated by 4; the network under it has the same layers without any dilation.
        """),
        mo.md(C.variants_table({"header": ("Network", "Last map", "Sees", "Circles", "Squares", "Triangles"),
                                "names": {"deeplab": "DeepLab", "same-rates": "DeepLab, ASPP 1, 1, 1",
                                          "dilated": "dilated, no ASPP", "plain": "no dilation, no ASPP",
                                          "fcn-4": "FCN, stride 4", "fcn-8": "FCN, stride 8", "fcn-16": "FCN, stride 16"}}, L.lang)),
        mo.md(r"""
        A finer map helps once every place sees enough. Each of the four networks with a last map of 32 by 32 whose places see 34 pixels or more has a higher median IoU than the FCNs with stride 8 and 16, for every kind; the FCN with stride 4 also has a map of 32 by 32, but its places see only 18 pixels, and it scores lowest of all. Dilation on its own does not help on the course's scenes: without an ASPP, the network with dilation has a lower median IoU than the one without, for every kind, and which of the two does better changes with the seed. A network that learns from scratch has no trained weights to keep: whatever spacing its taps have, it learns weights for it. An ASPP after the dilated conv4 lifts the network above both. But with dilation 1, 1 and 1, that ASPP does about as well as with 2, 4 and 8, so most of what the ASPP adds here comes from its extra branches, and with them almost twice as many parameters: 193,188 against 98,724. The dilations 2, 4 and 8 do give the circles a higher IoU with every seed. What DeepLab first used dilation for, the third cell of the worked example showed: a network trained on photos keeps its weights when its poolings come out. The DeepLab papers also use it to let a place see far around it on large photos without pooling; on the course's scenes, that did not pay off.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Semantic segmentation with DeepLab:

    1. Start from a CNN with poolings, here the layers of the course's FCN with stride 16.
    2. Take out the last poolings, here the third and the fourth: the last map stays 4 times smaller than the scene instead of 16.
    3. Dilate every layer after a pooling taken out: by 2 after one, by 4 after two, with padding as large as the dilation.
    4. At the end, an ASPP: a 1×1 convolution and 3×3 convolutions with several dilations, side by side on the same map, each followed by ReLU; their maps stacked along the channels; a 1×1 convolution with ReLU back to `c` channels.
    5. A 1×1 convolution to a score for every class, and a transposed convolution that makes the scores as large as the scene.
    6. Train with the cross-entropy of every pixel, and measure the IoU of every kind, as the FCN.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe with layers that have not learned anything yet. The first cell makes one dilated convolution, and a 5×5 convolution whose kernel holds the same 9 weights spread out, with zeros between them: the two give the same map. The second cell stacks 3×3 convolutions with dilation 1, 2, 4 and 8 on a picture with a single 1 in its middle, every weight 1, and counts after each how many pixels the 1 reaches. The third cell does steps 2 and 3 for conv4 on the scene of [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), made again with `scene(18)`: once with the third pooling, as in the FCN, and once with that pooling taken out and conv4 dilated, with the same weights. Steps 4 and 5 are exercises 2 and 3, and step 6 is what the demo does. Run the cells as they are.
    """)
    return


@app.cell
def example_spread():
    import torch
    from torch import nn

    # a 3x3 convolution with dilation 2, and a 5x5 convolution with the same kernel spread out,
    # zeros between its taps
    torch.manual_seed(0)
    dilated = nn.Conv2d(1, 1, 3, padding=2, dilation=2, bias=False)
    spread = nn.Conv2d(1, 1, 5, padding=2, bias=False)
    kernel = torch.zeros(5, 5)
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                kernel[2 * i, 2 * j] = dilated.weight[0, 0, i, j]
        spread.weight[0, 0] = kernel
    print(kernel)

    pic = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        a = dilated(pic)
        b = spread(pic)
    print(a.shape, b.shape)
    print(round(float((a - b).abs().max()), 5))
    return nn, torch


@app.cell
def example_seen(nn, torch):
    # a stack of 3x3 convolutions with dilation 1, 2, 4 and 8, every weight 1, on a picture
    # with a single 1 in its middle: the pixels the 1 reaches see it
    dot = torch.zeros(1, 1, 41, 41)
    dot[0, 0, 20, 20] = 1
    for d in [1, 2, 4, 8]:
        conv = nn.Conv2d(1, 1, 3, padding=d, dilation=d, bias=False)
        nn.init.ones_(conv.weight)
        with torch.no_grad():
            dot = conv(dot)
        # the pixels above 0 in the middle row, and in the whole picture
        print(d, int((dot[0, 0, 20] > 0).sum()), int((dot[0, 0] > 0).sum()))
    return


@app.cell
def example_pooling(nn, torch):
    from everything_ai.scenes import scene

    # the layers of the course's FCN with stride 16, up to conv4, and the scene up to conv3
    torch.manual_seed(0)
    conv1 = nn.Conv2d(3, 16, 3, padding=1)
    conv2 = nn.Conv2d(16, 32, 3, padding=1)
    conv3 = nn.Conv2d(32, 64, 3, padding=1)
    conv4 = nn.Conv2d(64, 64, 3, padding=1)
    pool = nn.MaxPool2d(2)
    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        y = pool(torch.relu(conv2(pool(torch.relu(conv1(x))))))
        y = torch.relu(conv3(y))

        # the FCN: the third pooling, then conv4
        fcn = torch.relu(conv4(pool(y)))

        # 2. and 3. the pooling taken out: one that moves 1 place at a time, and conv4
        #    dilated by 2, with the same weights
        conv4d = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        conv4d.load_state_dict(conv4.state_dict())
        deeplab = torch.relu(conv4d(nn.MaxPool2d(2, stride=1)(y)))
    print(fcn.shape, deeplab.shape)
    # every second place of DeepLab's map, against the FCN's map
    print(round(float((deeplab[:, :, ::2, ::2] - fcn).abs().max()), 5))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the first cell, the 5×5 kernel has the 9 weights of the dilated one in every second row and column, and zeros between them. The two maps have the same shape, (1, 1, 8, 8), thanks to padding 2, and their largest difference is 0.0: a dilated convolution is a larger kernel with gaps. In the second cell, the 1 reaches 3 pixels of the middle row after the first layer, 7 after the second, 15 after the third and 31 after the fourth: 9, 49, 225 and 961 pixels of the picture, every pixel of a square of 3, 7, 15 and 31, without a gap. In the third cell, the FCN's map after conv4 is (1, 64, 16, 16). With the pooling that moves 1 place at a time, the map of 32 by 32 loses one place, and DeepLab's map is (1, 64, 31, 31). Its places 0, 2, 4 and so on to 30, 16 of them along each side, are the FCN's map: the largest difference is 0.0. Between them, DeepLab has 15 new places along each side, which the FCN never computed. The course's DeepLab of exercise 3 leaves the pooling out altogether, so its maps stay 32 by 32.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_field():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What one place sees

        Complete `field(layers)`: `layers` is a list with one tuple `(kernel, stride, dilation)` for every layer, from the scene up. A 3×3 convolution is `(3, 1, 1)`, with dilation 2 `(3, 1, 2)`, a pooling of 2 `(2, 2, 1)`, a 1×1 convolution `(1, 1, 1)`. It gives back how many pixels of the scene one place after the last layer sees, along one side. Start with one pixel, and a step of 1 between neighbors; every layer adds `(k - 1) * d` times the step, and then multiplies the step by its stride, as in [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en). The check tries the course's FCN with stride 8, three dilated convolutions, every branch of the course's DeepLab, a stack with stride 2 in both layers, and no layers at all.
        """),
        L.task("Make `field` count the pixels one place sees."),
    ])
    return


@app.cell(hide_code=True)
def exercise_field():
    box_field = L.code_box("field", lines=10)
    box_field
    return (box_field,)


@app.cell(hide_code=True)
def check_field(box_field):
    result_field = L.check("field", box_field.value)
    result_field
    return (result_field,)


@app.cell(hide_code=True)
def hints_field():
    L.hints(
        "field",
        [
            "Two names keep the count: `seen`, the pixels one place sees so far, and `step`, how many pixels of the scene two neighboring places of the map are apart so far. Without any layer, both are 1. Then a loop over the layers changes both.",
            "`for k, stride, d in layers:` gives every tuple's three numbers a name. In the loop, first `seen += (k - 1) * d * step`, then `step *= stride`.",
            "`seen = 1` and `step = 1`, the loop of hint 2, and after the loop `return ___`.",
        ],
        why="A layer works on its input, whose neighbors are `step` pixels apart, so the `(k - 1) * d` places it adds on top of the middle one are `(k - 1) * d * step` pixels. Its stride only counts for the layers after it, so `step` changes after `seen`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_aspp():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ASPP

        Complete the class `ASPP(c_in, c, rates)`, step 4 of the recipe. In `__init__`: `self.branches`, a `nn.ModuleList` with first a 1×1 convolution from `c_in` channels to `c`, and then a 3×3 convolution from `c_in` to `c` for every dilation `r` in `rates`, with `padding=r` and `dilation=r`; and `self.project`, a 1×1 convolution that gets the `c` channels of every branch and gives `c`. In `forward`: every branch on the same `x`, each followed by ReLU; their maps stacked along the channels with `torch.cat(..., dim=1)`; and `self.project`, followed by ReLU. The check tries an ASPP with the dilations 2 and 4 on random maps, counts its parameters, and tries another with three dilations on a map that is not square. When it is right, it saves `ASPP` in your project, as `work/projects/cnn/aspp.py`; your DeepLab builds on it.
        """),
        L.task("Make `ASPP` put a 1×1 and dilated 3×3 convolutions side by side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_aspp():
    box_aspp = L.code_box("aspp", lines=16)
    box_aspp
    return (box_aspp,)


@app.cell(hide_code=True)
def check_aspp(box_aspp):
    result_aspp = L.check("aspp", box_aspp.value)
    result_aspp
    return (result_aspp,)


@app.cell(hide_code=True)
def hints_aspp():
    L.hints(
        "aspp",
        [
            "`self.branches` holds the convolutions side by side, like a list, but in a `nn.ModuleList`, so that PyTorch sees their weights. A list comprehension can make one 3×3 convolution for every `r` in `rates`; a list with the 1×1 convolution goes before it, with `+`.",
            "`nn.ModuleList([nn.Conv2d(c_in, c, 1)] + [nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates])`. With the 1×1 convolution, there are `len(rates) + 1` branches, so `self.project = nn.Conv2d(c * (len(rates) + 1), c, 1)`.",
            "In `forward`: `maps = [torch.relu(branch(x)) for branch in self.branches]`, then `return ___`, with the maps stacked along `dim=1`, `self.project` and ReLU.",
        ],
        why="Every branch gets the same map, and `padding=r` keeps the map of the branch with dilation `r` that size, so every branch gives a map of the same size, as `torch.cat` needs. Stacked along the channels, `self.project` gets every branch whole, and learns how much of each to use at every place.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_deeplab():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your DeepLab

        Complete the class `DeepLab`. `__init__` already has conv1, conv2 and conv3 of the course's FCN with stride 16, the pooling, your `ASPP` with the dilations 2, 4 and 8, the 1×1 convolution to the scores, and your `upsampler` of the last lesson with stride 4. Fill in `self.conv4`: a 3×3 convolution from 64 channels to 64. In the FCN, it came after the third pooling, which is gone. Then `forward`: conv1 and conv2, each followed by ReLU and the pooling; conv3 and conv4, each followed by ReLU, without pooling; then the ASPP, the scores and the upsampler. The check changes one pixel of a scene of 128 by 128 and counts the rows of the scores it reaches, tries two random pictures of 32 by 32 and counts the parameters, and tries a picture of 24 by 40. When it is right, it saves `DeepLab` in your project, as `work/projects/cnn/deeplab.py`; the demo trains it.
        """),
        L.task("Make `DeepLab` pool twice, and dilate what comes after."),
    ])
    return


@app.cell(hide_code=True)
def exercise_deeplab():
    box_deeplab = L.code_box("deeplab", lines=26)
    box_deeplab
    return (box_deeplab,)


@app.cell(hide_code=True)
def check_deeplab(box_deeplab):
    result_deeplab = L.check("deeplab", box_deeplab.value)
    result_deeplab
    return (result_deeplab,)


@app.cell(hide_code=True)
def hints_deeplab():
    L.hints(
        "deeplab",
        [
            "conv4 came after one pooling that is gone, so it gets dilation 2, as conv4d in the third cell of the worked example, with padding 2 so that the map keeps its size. Only conv1 and conv2 are followed by `self.pool`.",
            "`self.conv4 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)`. In `forward`: `x = self.pool(torch.relu(self.conv1(x)))`, the same for conv2, then `x = torch.relu(self.conv3(x))` and the same for conv4.",
            "After conv4: `return ___`, with the ASPP first, then the scores, then the upsampler.",
        ],
        why="Without the third and the fourth pooling, the map stays 4 times smaller than the scene, and the upsampler with stride 4 makes the scores as large as the scene. conv4 and the ASPP get dilations, so that they reach about as far as they would have after those poolings.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Press the button: the demo trains your `DeepLab`, with your `ASPP` and your `upsampler`, on the class maps your `class_map` makes of the 2,000 training scenes, for 8 epochs in mini-batches of 32, just as the course's FCN trained. After every epoch, it measures your DeepLab on the 200 test scenes with your `kind_iou`. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; on the course's computer, that took about 6 minutes. The chart shows the loss on the test scenes after every epoch. When it is done, you can go back through the epochs, and see the classes your DeepLab gave three test scenes after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your DeepLab")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_deeplab(train_button, result_aspp, result_deeplab):
    result_aspp, result_deeplab  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your DeepLab.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your DeepLab: the loss on the test scenes", "x": "epoch", "y": "test loss",
                                        "alt": "The loss of your DeepLab on the 200 test scenes after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Your DeepLab** · epoch {_p['epoch']} of {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:
                _now["chart"].line(f"**Your DeepLab** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercise 3 first: then your `DeepLab` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than twenty minutes. Does your DeepLab do more work than it needs, such as the second pooling taken out too, or more channels than the recipe? If it follows the recipe, this computer is too slow for the demo.*",
                "classes": "*Your `class_map` in `work/projects/cnn/classes.py` stops with an error, or does not give a class from 0 to 3 for every pixel of a scene. Did you change its file? Delete it, and check its exercise in [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) again.*",
                "deeplab": "*Your `DeepLab` in `work/projects/cnn/deeplab.py`, or the `ASPP` or the `upsampler` it uses, stops with an error, does not give 4 scores for every pixel of a scene, or trains to a loss that is not a number. Did you change one of their files? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
                "kind_iou": "*Your `kind_iou` in `work/projects/cnn/kind_iou.py` stops with an error, or does not give 1 for a class map compared with itself. Did you change its file? Delete it, and check its exercise in [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "The loss of your DeepLab after every epoch: on the training scenes solid, on the test scenes dashed."}
            _words = {"scene": "scene {seed}", "kinds": ("circle", "square", "triangle"),
                      "alt": "Three test scenes, each with the class of every pixel that your DeepLab gave it after that epoch."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C._FCN.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C._FCN.rewind_picture(_moments[_e - 1], _words),
                        f"After epoch {_e}: test loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: which kind does your DeepLab find last?")]
            else:
                _pic = [mo.Html(C._FCN.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _ious = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("circles", "squares", "triangles"), _result["ious"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Your DeepLab, with {C._FCN.whole(_result['params'], L.lang)} parameters, gets {L.number(100 * _result['acc'], 1)}% of the pixels of the 200 test scenes right, and the IoU of the pixels of every kind, by your `kind_iou`, is: {_ious}. With seed 0, the run the demo follows, the course's DeepLab gets 98.8% of the pixels right, with an IoU of 0.91 for the circles, 0.81 for the squares and 0.82 for the triangles; over the seeds 0 to 4, the table after the slider gives the lowest and the highest.{_who}"""),
                mo.md("**Your DeepLab, layer by layer.** The first box is the scene of the worked example, and the boxes after it are the layers of your trained DeepLab, in the order the scene goes through them; under them are the feature maps of the layer you choose."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Go through your trained DeepLab from the scene to the scores: from which layer on does the map stop getting smaller?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
                mo.md("After the second pooling, layer 4, the map stays 32 by 32: conv3, conv4 and your ASPP keep it as large as it was, with padding as large as their dilation, and so do the scores, 4 channels for the 4 classes. In the FCN with stride 16, a third and a fourth pooling made it 16 by 16 and then 8 by 8. Your ASPP is one box: its four branches work inside it, side by side. Only the upsampler, the last layer, makes the map larger again: 128 by 128, as large as the scene."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 4. Why dilate
    """)
    return


@app.cell(hide_code=True)
def explain_dilate_cell():
    explain_dilate = L.explain(
        "why-dilate",
        question="The first DeepLab, of 2015, starts from a VGG-16 trained on ImageNet, sets the stride of its last two poolings to 1, and dilates the layers after them. Why the dilation? What would go wrong if it left those layers as they were? And why did dilation on its own not help the course's networks, which learn from scratch?",
        model_answer="Every layer after a pooling learned its weights for neighbors that are one place apart in its input, and so two places apart in the map before that pooling. Take the pooling out and leave the layer as it is, and its taps land on neighbors twice as close: it multiplies other values than it learned for, and sees half as far. With dilation 2, its taps are two places apart again and reach exactly the values they reached before: at every other place, it gives the same numbers as before, as the third cell of the worked example showed, so its trained weights keep working while the map stays twice as large. The course's networks learn from scratch, so they have no trained weights to keep: whatever spacing their taps have, they learn weights for it. Dilation then only lets them see a larger part of the scene, and on the course's scenes that did not help: without an ASPP, the network with dilation did a little worse at the median than the one without.",
        points=[
            "A layer that learned after a pooling has its weights for neighbors that are 2 places apart in the map before that pooling.",
            "With the pooling taken out and dilation 2, its taps land on exactly those places again: at every other place, it gives what it gave before, so its weights keep working.",
            "Trained from scratch, a network has no weights to keep: it learns weights for whatever spacing its taps have, so dilation only lets it see further, and on the course's scenes that did not help.",
        ],
    )
    explain_dilate
    return (explain_dilate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** What one place sees, from [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), now counts dilation too. The layers of the course's FCN from [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) are now a DeepLab with two of their poolings taken out, ended by your `upsampler` from that lesson and measured with your `kind_iou`; the 1×1 convolution of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en) now combines the branches of an ASPP. On the course's scenes, a finer map whose places see enough and the ASPP are what help: over five seeds, the course's DeepLab reaches an IoU of 0.91 to 0.93 for the circles, where the FCN with stride 8 reached 0.78 to 0.87. What DeepLab first used dilation for is keeping the weights a network learned on photos. Your project `cnn` now has `aspp.py` and `deeplab.py`. In the next lesson, *A mask for every object: the idea of Mask R-CNN*, a detector gives every shape it finds a mask of its own, so that two shapes of the same kind are no longer one patch of pixels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_field, result_aspp, result_deeplab, explain_dilate):
    L.wrap_up(result_field, result_aspp, result_deeplab, explain_dilate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
