# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-efficientnet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-efficientnet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Depthwise separable convolutions: MobileNet](/course/open/cv-mobilenet?lang=en) and [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_separable():
    quick_separable = L.choice(
        "quick-separable",
        question="`SeparableConv(64, 64)` needs 9 × 64 + 64 × 64 = 4,672 multiplications for each pixel of its maps: 9 for each channel in its depthwise convolution, and 64 for each channel in its 1×1 convolution. How many does `SeparableConv(128, 128)` need for each pixel?",
        options={
            "right": "17,536: almost 4 times as many, because the 1×1 convolution goes from 64 × 64 to 128 × 128",
            "twice": "9,344: twice as many, because there are twice as many channels",
            "same": "4,672: its filters are still 3 by 3 and 1 by 1",
        },
        correct="right",
        why={
            "right": "9 × 128 + 128 × 128 = 1,152 + 16,384 = 17,536, 3.75 times as many. The 1×1 convolution grows with the channels that come in times the channels that go out: twice as many of both is 4 times as much. Today you count what a wider network costs, and a deeper one, and one that gets larger photos.",
            "twice": "The depthwise convolution doubles, to 9 × 128, but the 1×1 convolution gets twice as many filters, each twice as deep: 128 × 128 = 16,384, 4 times 64 × 64. See [MobileNet](/course/open/cv-mobilenet?lang=en).",
            "same": "The filters keep their size, but there are more of them, and each 1×1 filter is as deep as the channels it gets. See [MobileNet](/course/open/cv-mobilenet?lang=en).",
        },
    )
    quick_separable
    return (quick_separable,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="A block with a residual connection gives `x + f(x)`, where `f` is the block's own layers. What must `f(x)` give back for that to work?",
        options={
            "right": "As many channels as `x`, on maps of the same size: `+` adds them number by number",
            "half": "Half as many channels as `x`: the block makes them fewer, as a bottleneck does",
            "any": "Any number of channels: PyTorch adds up what it can",
        },
        correct="right",
        why={
            "right": "The sum has the shape of `x`, so `f(x)` must have it too. Today every block is a depthwise separable convolution from `width` channels to `width`, whose output is added to `x`.",
            "half": "Then `x + f(x)` adds maps of 32 channels to maps of 64, and PyTorch stops. A bottleneck makes the channels fewer in its middle, and as many again at its end. See [ResNet](/course/open/cnn-resnet?lang=en).",
            "any": "PyTorch stops when the channels or the size of the maps do not fit: the block must keep both. See [ResNet](/course/open/cnn-resnet?lang=en).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A larger CNN usually reads more photos right, but it also needs more computing power. When you have more, how should you spend it? ResNet grows from ResNet-18 to ResNet-200 by adding layers: deeper. Wide ResNets give every layer more channels: wider. GPipe made a network four times as large and gave it photos of 480 by 480 pixels: sharper. In 2019, Mingxing Tan and Quoc Le at Google measured what each of the three gains for what it costs, and found that growing all three together, in a fixed ratio, does better than growing one: **compound scaling**. They grew a small network, EfficientNet-B0, step by step into EfficientNet-B1 to B7. EfficientNet-B7 got 84.3% of ImageNet's photos right, as many as GPipe, with 66 million parameters against 557 million: 8.4 times fewer. Today you count what each way of growing costs, build a network that grows, and test on CIFAR-10 which way pays off there.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_grown():
    guess_grown = L.guess(
        "grown-imagenet",
        question="Tan and Le grew EfficientNet-B0, which got 77.3% of ImageNet's photos right, to about 4.5 times its multiplications in four ways: only deeper, only wider, only sharper photos, or all three a little. Which did best?",
        options={
            "same": "All four about the same: the same multiplications buy about the same",
            "sharper": "Only sharper photos, by far: larger photos show more",
            "right": "All three a little, by about 2 points over each of the others",
        },
        reveal="**All three a little: 81.1%.** Only deeper got 79.0% right, only wider 78.9%, only sharper 79.1%, each with 1.8 to 1.9 billion multiplications for a photo, against 0.4 billion for B0. All three together got 81.1%, with 1.8 billion (the paper's table 7). Each way alone soon gains less and less; together they help each other.",
    )
    guess_grown
    return (guess_grown,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Three ways to grow.** A CNN can grow in three directions. Its **depth** is how many blocks it has after each other; its **width** is how many channels each block has, side by side; and the **resolution** of its photos is how many pixels they have, so how many rows and columns each map has. A network that gets photos of a higher resolution is called sharper here: it sees finer detail. The picture: the base network of today's demo, and four ways to grow it. At the bottom of each, the photo it gets, drawn to scale: the more pixels, the larger; above it, a bar for each block, as wide as its channels. Under each, how many times the base network's multiplications it needs.
        """),
        C.networks_picture({"base": "base", "deeper": "deeper", "wider": "wider", "sharper": "sharper", "compound": "all three",
                            "blocks": "{d} blocks", "channels": "{w} channels", "photo": "{s} × {s} pixels", "times": "× {t}",
                            "alt": "Five networks side by side, each a gray square for its photo with a bar for each block above it, as wide as its channels. Base: 2 blocks of 64 channels on photos of 16 by 16, × 1.00. Deeper: 9 blocks of 64 channels, 16 by 16, × 3.95. Wider: 2 blocks of 136 channels, 16 by 16, × 3.89. Sharper: 2 blocks of 64 channels on photos of 32 by 32, × 4.00. All three: 3 blocks of 80 channels on photos of 22 by 22, × 4.01."}, L.lang),
        mo.md(r"""
        **What each costs.** A convolution needs one multiplication for each weight of a filter, for every number it gives. Twice as many blocks: twice the multiplications. Twice as many channels: each filter sees twice as many channels, and there are twice as many filters, so 4 times. Photos twice as wide and twice as high: every map has 4 times as many pixels, so 4 times as well. The paper writes it as a rule: the multiplications grow with

        $$d \cdot w^2 \cdot r^2,$$

        where $d$, $w$ and $r$ say how many times as deep, as wide and as sharp the network has become; $r$ multiplies both the height and the width of the photo, so its pixels grow with $r^2$. The parameters grow differently: with the depth, and with the square of the width, but not with the resolution. On a larger photo, the same filters slide over more pixels.

        **Compound scaling.** Tan and Le grow all three together, with one number $\varphi$ (phi) that says how much:

        $$d = \alpha^\varphi, \qquad w = \beta^\varphi, \qquad r = \gamma^\varphi, \qquad \text{with } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2.$$

        By the rule, the multiplications then grow with $(\alpha \cdot \beta^2 \cdot \gamma^2)^\varphi$: every step of $\varphi$ about doubles them. With $\varphi = 1$, they tried a few values of $\alpha$, $\beta$ and $\gamma$ on EfficientNet-B0, and kept the best: $\alpha = 1.2$, $\beta = 1.1$ and $\gamma = 1.15$, with $\alpha \cdot \beta^2 \cdot \gamma^2 = 1.92$. Then they kept those three and took a larger $\varphi$ for each of EfficientNet-B1 to B7. Why all three together? A larger photo needs more layers, so that a filter late in the network still sees enough of it, its receptive field; and more channels, for the finer patterns the extra pixels show.

        **EfficientNet-B0** itself was found by a computer search through many small networks, built from depthwise convolutions and 1×1 convolutions as in MobileNet, for the one that reads the most photos right for its multiplications. The paper counts in FLOPS, a multiplication and its addition counted as one; here they are the multiplications. Its table 7 grows B0 in each of the four ways to about 4.5 times its multiplications:

        | EfficientNet-B0 | Multiplications for a photo | Right on ImageNet |
        |---|---|---|
        | as it is | 0.4 billion | 77.3% |
        | deeper, $d = 4$ | 1.8 billion | 79.0% |
        | wider, $w = 2$ | 1.8 billion | 78.9% |
        | sharper, $r = 2$ | 1.9 billion | 79.1% |
        | all three, $d = 1.4$, $w = 1.2$, $r = 1.3$ | 1.8 billion | 81.1% |

        The same holds for networks that were not made for it; its table 3, each grown to 3.6 to 4.3 times its multiplications:

        | Network | As it is | Deeper | Wider | Sharper | All three |
        |---|---|---|---|---|---|
        | MobileNet | 70.6% | — | 74.2% | 72.7% | 75.6% |
        | MobileNetV2 | 72.0% | 76.8% | 76.4% | 74.8% | 77.4% |
        | ResNet-50 | 76.0% | 78.1% | 77.7% | 77.5% | 78.8% |

        And the family, from its table 2, next to networks that get about as many photos right. B0 has 77.1% there, a little less than in table 7:

        | Network | Right on ImageNet | Parameters | Multiplications for a photo |
        |---|---|---|---|
        | EfficientNet-B0 | 77.1% | 5.3 million | 0.39 billion |
        | ResNet-50 | 76.0% | 26 million | 4.1 billion |
        | EfficientNet-B1 | 79.1% | 7.8 million | 0.70 billion |
        | ResNet-152 | 77.8% | 60 million | 11 billion |
        | EfficientNet-B4 | 82.9% | 19 million | 4.2 billion |
        | EfficientNet-B7 | 84.3% | 66 million | 37 billion |
        | GPipe | 84.3% | 557 million | not in the table |
        """),
        L.watch_out("The rule $d \\cdot w^2 \\cdot r^2$ is about right, not exact. The paper uses it because ordinary convolutions, the 1×1 convolutions among them, do most of the work in a CNN; but layers that grow another way count too, such as the first convolution and the depthwise convolutions, and so does rounding, as the worked example and the slider show. How much compound scaling gains also depends on the network you start from, the paper says: that is why Tan and Le searched for B0 first. And a higher resolution only helps when the photos have the detail. CIFAR-10's photos are 32 by 32, so the demo's base network gets them shrunk to 16 by 16, and the sharper networks get more of their pixels back, up to 32 by 32."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **depth**, **width**, **resolution** | how many blocks a network has after each other, how many channels each has, and how many pixels its photos have | `depth`, `width`, `side` |
        | **compound scaling** | growing a network in depth, width and resolution together, each by a fixed factor to the power $\varphi$ | — |
        | **EfficientNet** | a family of CNNs from 2019: B0, found by a computer search, and B1 to B7, grown from it with compound scaling | — |
        | $d$, $w$, $r$ | how many times as deep, as wide and as sharp a grown network is as the one it grew from | — |
        | $\varphi$; $\alpha$, $\beta$, $\gamma$ | the one number that says how much to grow; and the factors for depth, width and resolution in each step of it | `phi` |

        **The demo's network.** `ScaledCNN(depth, width)` starts with a 3×3 convolution with stride 2 from the 3 colors to `width` channels, which makes the maps half as wide and high. Then come `depth` blocks, each your depthwise separable convolution of [MobileNet](/course/open/cv-mobilenet?lang=en), `SeparableConv(width, width)`, with a residual connection as in [ResNet](/course/open/cnn-resnet?lang=en). At the end, it takes the mean of each map, `width` numbers whatever the size of the photo, and a linear layer makes the 10 logits from them. The base network has 2 blocks of 64 channels, and gets the photos of CIFAR-10 made 16 by 16 by the course, with the interpolation of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en). It has 12,042 parameters, and needs 709,248 multiplications for a photo. The four grown networks each need about 4 times as many. Deeper: 9 blocks. Wider: 136 channels. Sharper: CIFAR-10's photos as they are, 32 by 32. All three: the paper's $\alpha$, $\beta$ and $\gamma$ with $\varphi = 2$, rounded: 3 blocks, 80 channels and photos of 22 by 22. By the rule, 4 times the blocks would be 8 and twice the channels 128; counted exactly, those cost only 3.53 and 3.48 times as much, so the course takes 9 and 136. The worked example shows why.

        The picture: how many of the first 1,000 test photos of CIFAR-10 each network reads right after 20 epochs on the first 6,000 training photos, a dot for each of 5 seeds, with the course's code. The base network: 506 to 524. Deeper: 513 to 536. Wider: 536 to 555. Sharper: 514 to 558. All three together: 543 to 571. Exercise 3 asks what the rows say.
        """),
        mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                "base": "base: 2 blocks, 64 channels, 16 × 16", "deeper": "deeper: 9 blocks",
                                "wider": "wider: 136 channels", "sharper": "sharper: photos of 32 × 32",
                                "compound": "all three: 3 blocks, 80 channels, 22 × 22", "mine": "your run", "course": "the course's runs",
                                "alt": "Five rows of open dots, test photos read right of 1,000, one dot for each seed and a short line at their mean. Base network: 506 to 524. Deeper: 513 to 536. Wider: 536 to 555. Sharper: 514 to 558. All three together: 543 to 571."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a network that grows in depth and width:

    1. In `__init__`, make `depth` blocks `SeparableConv(width, width)`, a new one for every place, and keep them in an `nn.ModuleList`, so that PyTorch sees their parameters.
    2. In `forward`, after the first convolution and its ReLU, go through the blocks in order, and add what each gives to `x`: `x = x + block(x)`.
    3. The mean of each map and the linear layer after it make the logits; they are already in the box.

    The resolution is not in the network: it is the size of the photos you give it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell counts the multiplications of `ScaledCNN(depth, width)` for one photo of `side` by `side` pixels, layer by layer, as the demo's network has them. It prints those of the base network, those of a grown one, and how many times as many that is. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_cost():
    def multiplications(depth, width, side):
        half = side // 2  # the first convolution has stride 2 (an even side)
        first = 3 * 3 * 3 * width * half * half
        # a block: 9 for each channel, then width for each channel, at every pixel
        block = (3 * 3 * width + width * width) * half * half
        last = width * 10  # the linear layer, from the means of the maps
        return first + depth * block + last

    base = multiplications(2, 64, 16)
    depth, width, side = 8, 64, 16  # change: depth, width, side = 2, 128, 16
    grown = multiplications(depth, width, side)
    print(base, grown, grown / base)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `709248 2503296 3.529507309149973`: 4 times as many blocks cost only 3.53 times as much. The blocks do grow 4 times, from 598,016 to 2,392,064, but the first convolution, 110,592, and the linear layer, 640, stay as they are. With `depth, width, side = 2, 128, 16`, it prints `709248 2467072 3.4784334957588885`: twice the channels cost 3.48 times as much. The 1×1 convolutions in the blocks grow 4 times, but the depthwise convolutions and the first convolution only 2 times, because each of their filters sees one channel or the 3 colors. Try `2, 64, 32` as well: then every layer but the linear one gets 4 times the pixels, and the cell prints `709248 2835072 3.9972929074174335`, almost exactly 4.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide $\varphi$ from 0 to 5: the base network grown with compound scaling and the paper's $\alpha$, $\beta$ and $\gamma$, rounded as in the demo: the depth up to whole blocks, the width to a multiple of 8, the side to an even number. The bars are the multiplications, counted, as a multiple of the base network's, on a log scale, where every doubling is a step of the same height; dark, the $\varphi$ you chose. The dashed line is what the rule says: 1.92 times as many for every step, so a straight line.
    """)
    return


@app.cell(hide_code=True)
def play_phi():
    _labels = {"title": "The base network grown with compound scaling", "x": "φ", "y": "times the base network's",
               "rule": "the rule: 1.92 per step",
               "alt": "Six bars, the multiplications of the base network grown with compound scaling for φ from 0 to 5, as a multiple of the base network's, counted, on a log scale, with the bar of the chosen φ dark, and a dashed line with open dots for what the rule says, 1.92 to the power φ."}
    explore_phi = L.explore("phi-slider", "φ", lambda _v: (C.family_picture(int(_v), _labels, L.lang), C.family_sentence(int(_v), L.lang)),
                            start=2, values=C.PHIS, role="hyper",
                            task="Slide φ from 0 to 5. At which φ does the counted number first fall below what the rule says?")
    explore_phi
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At $\varphi = 5$: 22.22 times the base network's multiplications, where the rule says 26.11. Two things work against each other. The rounding, above all of the depth, makes the network larger than the rule says: at $\varphi = 1$ it already has 3 blocks, where the rule has 2.4. And the first convolution and the depthwise convolutions grow only with the width, not with its square, which makes it smaller, the more so the wider it gets. From $\varphi = 1$ to 4 the rounding wins; at $\varphi = 5$ the depth rounds from $2 \cdot 1.2^5 = 4.98$ to 5 blocks, hardly up, and the width wins. There, too, the photos are as large as CIFAR-10 has them: 32 by 32. A larger $\varphi$ would need photos that CIFAR-10 does not have. Each step about doubles the cost, as the rule says, but never exactly.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_costs():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What each choice costs

    Eight ways to grow a network. Which give it about 2 times as many multiplications, by the rule $d \cdot w^2 \cdot r^2$, and which about 4 times? Sort each one.
    """)
    return


@app.cell(hide_code=True)
def exercise_costs():
    sort_costs = L.sort("costs")
    sort_costs
    return (sort_costs,)


@app.cell(hide_code=True)
def check_costs(sort_costs):
    result_costs = L.check("costs", sort_costs)
    result_costs
    return (result_costs,)


@app.cell(hide_code=True)
def hints_costs():
    L.hints(
        "costs",
        [
            "The depth counts once, the width and the resolution twice: $d \\cdot w^2 \\cdot r^2$. What is 2 squared, and what is 1.41 squared?",
            "Four go in each group. One step of $\\varphi$ multiplies by $\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 = 1.92$, two steps by $1.92^2 = 3.69$.",
        ],
        why="**About 2 times:** depth × 2, width × 1.41, resolution × 1.41, and $\\varphi$ one higher (1.92). **About 4 times:** width × 2, resolution × 2, depth × 4, and $\\varphi$ two higher (3.69). Width and resolution count squared, the depth once.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scaled():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A network that grows

        Complete `ScaledCNN`. Its arguments are the number of blocks, `depth`, and the channels of each, `width`. The first convolution, the linear layer and the end of `forward` are in the box. Make the blocks: `depth` of your `SeparableConv` from `width` channels to `width`, in an `nn.ModuleList`. Then, in `forward`, send `x` through each block in turn, with a residual connection. The check builds your network with 2 blocks of 64 channels and with other numbers, and compares what comes out, how many blocks and parameters it has, and whether every parameter gets a gradient, with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/scaled.py`.
        """),
        L.task("Make the blocks in `__init__`, and the loop over them in `forward`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scaled():
    box_scaled = L.code_box("scaled", lines=20)
    box_scaled
    return (box_scaled,)


@app.cell(hide_code=True)
def check_scaled(box_scaled):
    result_scaled = L.check("scaled", box_scaled.value)
    result_scaled
    return (result_scaled,)


@app.cell(hide_code=True)
def hints_scaled():
    L.hints(
        "scaled",
        [
            "Step 1 of the recipe: a list of `depth` new blocks, one for each place. How do you make such a list in one line, and what must go around it, so that PyTorch sees the blocks?",
            "`self.blocks = nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])`. In `forward`: `for block in self.blocks:` and, indented under it, step 2 of the recipe.",
        ],
        why="The blocks are `nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])`, and `forward` goes through them with `for block in self.blocks:` and `x = x + block(x)`. Most of EfficientNet's blocks first make the channels 6 times as many with a 1×1 convolution, before their depthwise convolution, and they use batch normalization; its depth, width and resolution grow just as here.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains your `ScaledCNN` as the network you choose on the first 6,000 training photos of CIFAR-10, made as large as that network needs, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Its blocks are your `SeparableConv`. Where a module of yours from an earlier lesson is missing or broken, the course's stands in. It needs CIFAR-10 and exercise 2, and takes a few minutes. While it trains, a chart shows the loss on the training photos every 10 batches, and dashed on the test photos after every epoch. Beside or under it are the network's guesses at that moment for one test photo of each kind, and under them the feature maps of its first layer on a car; both are drawn again as it trains. In the picture after, the gray line is the validation loss of the course's base network with seed 0. Then you can rewind to a few epochs of your run, with both pictures as they were then, and see your run next to the course's.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 is not on your computer yet. If you did [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once from the University of Toronto, 170 MB, unpacks its photos into one file of 184 MB, and then deletes the download. Its page states no license, so the course does not ship it; where its photos come from is in that project lesson. Only the demo needs it. While it downloads, the whole lesson waits.*",
        "busy": "**Downloading CIFAR-10** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of CIFAR-10** · this takes a few seconds",
        "no_net": "*The download did not work: the University of Toronto could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "damaged": "*Unpacking did not work: the downloaded file turned out to be damaged, so the course deleted it. Click again to download it again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 170 MB and the unpacked photos, 184 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*CIFAR-10 is on your computer: 50,000 training photos and 10,000 test photos.*",
    })
    return (cifar,)


@app.cell(hide_code=True)
def variant_choice_cell():
    variant_choice = mo.ui.radio(options={"all three together: 3 blocks, 80 channels, 22 × 22": "compound",
                                          "deeper: 9 blocks": "deeper", "wider: 136 channels": "wider",
                                          "sharper: photos of 32 × 32": "sharper",
                                          "the base network: 2 blocks, 64 channels, 16 × 16": "base"},
                                 value="all three together: 3 blocks, 80 channels, 22 × 22", label="The network")
    train_button = mo.ui.run_button(label="Train the network")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_scaled(cifar, train_button, variant_choice, result_scaled):
    cifar, result_scaled  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the network.*")
    else:
        _variant = variant_choice.value
        _names = {"base": "Base", "deeper": "Deeper", "wider": "Wider", "sharper": "Sharper", "compound": "All three"}
        _with = {"base": "The base network", "deeper": "The deeper network, with 9 blocks,", "wider": "The wider network, with 136 channels,",
                 "sharper": "The sharper network, on photos of 32 by 32,", "compound": "The network grown all three ways"}
        _now = {}   # the live chart, made at the first report
        _layer = "Layer {name}, the first of your ScaledCNN, on a test photo of a car"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your network on CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "The loss of the network so far: on the training photos every 10 batches, and dashed on the 1,000 test photos after every epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "training photos, every 10 batches"), ("loss", True, "test photos, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-efficientnet-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**{_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `ScaledCNN` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `ScaledCNN`, your `SeparableConv` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "scaled": "*Your `ScaledCNN` in `work/projects/cnn/scaled.py`, or the `SeparableConv` it uses in `work/projects/cnn/separable.py`, stops with an error, or does not give back what it should: 10 logits for each photo. Did you change one of those files? Delete it, and check its exercise again: its check then saves your code again.*",
                "cnn": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give what it should. Did you change its file? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "base": "base network",
                       "alt": "The loss of the network after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's base network with seed 0, which falls to 1.363 by the last epoch."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _layer, L.lang),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.SHOWN)} photos read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: when is the validation loss lowest, and what happens after?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Its validation loss was lowest after epoch **{_best}**." if _best < C.EPOCHS else f" Its validation loss was still falling after epoch {_best}.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""{_with[_variant]} reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters and {C.grouped(_result['mults'], L.lang)} multiplications for a photo.{_turn} Over 5 seeds, with the course's code, this network: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos, with the lowest validation loss after epoch {_be[0]} to {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                        "base": "base: 2 blocks, 64 channels, 16 × 16", "deeper": "deeper: 9 blocks",
                                        "wider": "wider: 136 channels", "sharper": "sharper: photos of 32 × 32",
                                        "compound": "all three: 3 blocks, 80 channels, 22 × 22", "mine": "your run", "course": "the course's runs",
                                        "alt": "Five rows of test photos read right, of 1,000: open dots for the course's runs over 5 seeds, for the base network, deeper, wider, sharper and all three together, with a short line at the mean of each row; and in the row of your network a filled dot for your run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Exercise 3. Which way pays off here
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "scaling-runs",
        question="Over 5 seeds, after 20 epochs, test photos read right of 1,000. The base network reads 506 to 524. Grown to about 4 times its multiplications: deeper, 513 to 536; wider, 536 to 555; sharper, 514 to 558; all three together, 543 to 571. What do the runs say?",
        options={
            "deeper": "Deeper is best: it learns the most of its training photos, 4,802 to 4,979 of 6,000",
            "sharper": "Sharper is best: its best run, 558, shows that larger photos are what this network lacks",
            "right": "All three together read the most: 558 on average, against 543 for wider, 534 for sharper and 525 for deeper, though its runs overlap with those of wider and sharper; deeper hardly differs from the base network",
        },
        correct="right",
        why={
            "deeper": "Deeper learns its training photos best, 4,802 to 4,979, but reads the fewest test photos of the four grown networks, 513 to 536: its validation loss is lowest after epoch 10 to 13, and then rises. It learns its training photos by heart, overfitting; the others have their lowest validation loss after epoch 15 to 20.",
            "sharper": "One run is not the row: sharper reads 514 to 558, the widest spread, 534 on average, below wider and all three together. It has no more parameters than the base network, 12,042, and learns no more of its training photos: 3,592 to 3,720, against 3,590 to 3,686. And with 2 blocks, a number of the last maps, before the mean, sees 11 by 11 pixels of a photo of 32 by 32. Larger photos need more layers and more channels, the paper says, which is why compound scaling grows them together; these runs do not test that.",
            "right": "All three together read 543 to 571 right, 558.4 on average; wider 542.6, sharper 533.8, deeper 524.8, and the base network 517.2. Each grown network needs about 4 times the base network's multiplications. As in the paper, all three together came first: there with 81.1% right, against 78.9 to 79.1% for one way. But with 5 seeds not every difference is sure: its lowest run, 543, is still below the best of wider, 555, and of sharper, 558, and deeper, 513 to 536, hardly differs from the base network, 506 to 524. A finding holds where it was measured: on 6,000 photos of CIFAR-10, with this network.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The blocks of your network are the depthwise separable convolutions of [Depthwise separable convolutions: MobileNet](/course/open/cv-mobilenet?lang=en), each with the residual connection of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en). You counted their multiplications with the shapes of [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), made the photos smaller with the interpolation of [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en), and trained with the Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) on the photos of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en). Your project `cnn` now has `scaled.py`. The next lesson, *Saliency: the pixels that change the answer*, asks another question of a CNN: not how large it should be, but which pixels of a photo made it choose its answer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_costs, result_scaled, choice_runs):
    L.wrap_up(result_costs, result_scaled, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
