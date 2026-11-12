# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-mobilenet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-mobilenet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Groups of channels: ResNeXt](/course/open/cv-resnext?lang=en) and [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_groups():
    quick_groups = L.choice(
        "quick-groups",
        question="`nn.Conv2d(64, 64, 3, padding=1, groups=4)` splits its 64 channels into 4 groups. How many weights does each of its filters have?",
        options={
            "right": "144: it sees the 16 channels of its own group, 16 × 3 × 3",
            "all": "576: it sees all 64 channels, 64 × 3 × 3",
            "four": "36: one filter of 3 × 3 for each of the 4 groups",
        },
        correct="right",
        why={
            "right": "With 4 groups, each group has 64 / 4 = 16 channels, and each filter sees only the channels of its group: 16 × 9 = 144 weights. Today the groups get as small as they can: one channel each.",
            "all": "That is a convolution without groups. With `groups=4`, a filter sees only the 16 channels of its own group. See [ResNeXt](/course/open/cv-resnext?lang=en).",
            "four": "A filter is as deep as the channels of its group, and a group has 64 / 4 = 16 of them. See [ResNeXt](/course/open/cv-resnext?lang=en).",
        },
    )
    quick_groups
    return (quick_groups,)


@app.cell(hide_code=True)
def quick_check_pointwise():
    quick_pointwise = L.choice(
        "quick-pointwise",
        question="A 1×1 convolution from 64 channels to 64 works out one number of its output at the pixel in row 2, column 3. Which numbers of its input does it use?",
        options={
            "right": "The 64 numbers of that pixel, one from each channel",
            "neighbors": "The 3 × 3 pixels around it, in every channel",
            "one": "Only the number of that pixel in one channel",
        },
        correct="right",
        why={
            "right": "A 1×1 filter is 64 numbers deep and 1 pixel wide: it mixes the channels of one pixel, and does not look at its neighbors. Today the 1×1 convolution is one of two steps that share the work of a 3×3 convolution.",
            "neighbors": "That is a 3×3 convolution. A 1×1 filter covers one pixel, in all its channels. See [1×1 convolutions](/course/open/cv-1x1-convolution?lang=en).",
            "one": "A 1×1 filter is as deep as the channels it gets: it uses all 64 numbers of the pixel. See [1×1 convolutions](/course/open/cv-1x1-convolution?lang=en).",
        },
    )
    quick_pointwise
    return (quick_pointwise,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A phone has little computing power and a battery that runs out. In 2017, Andrew Howard and his colleagues at Google built MobileNet for computers like that. A 3×3 convolution does two jobs at once: each filter looks at the neighbors of a pixel, and adds up what it finds in all the channels. MobileNet splits the two jobs: first a convolution that gives every channel a 3×3 filter of its own, the grouped convolution of [Groups of channels: ResNeXt](/course/open/cv-resnext?lang=en) with as many groups as channels; then a 1×1 convolution that mixes the channels, as in [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en). On ImageNet, MobileNet got 70.6% of the photos right with 4.2 million parameters and 569 million multiplications for a photo. The same network with ordinary 3×3 convolutions got 71.7% right, with 29.3 million parameters and 4,866 million multiplications; VGG-16, 71.5% with 138 million and 15,300 million. Today you count what the split saves, build it, and test in a CNN on CIFAR-10 what each of its two steps does.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_mults():
    guess_mults = L.guess(
        "separable-mults",
        question="A 3×3 convolution from 64 channels to 64, on maps of 8 by 8, gives 4,096 numbers: 64 channels of 64 pixels. Each needs 9 × 64 = 576 multiplications, 2,359,296 together. A depthwise convolution, a 3×3 filter for each channel on its own, with a 1×1 convolution from 64 channels to 64 after it, gives maps of the same shape. How many times fewer multiplications does it need?",
        options={
            "two": "About 2 times fewer",
            "right": "About 8 times fewer",
            "many": "About 64 times fewer",
        },
        reveal="**About 8 times: 7.89.** The depthwise convolution needs 9 multiplications for each of its 4,096 numbers, 36,864; the 1×1 convolution needs 64 for each, 262,144. Together 299,008, against 2,359,296. Nearly all of it is in the 1×1 convolution: the depthwise convolution looks at the neighbors almost for free.",
    )
    guess_mults
    return (guess_mults,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The depthwise convolution.** A depthwise convolution does only the first job of a 3×3 convolution: it gives every channel a 3×3 filter of its own, which sees that channel and no other, and makes one map from it. In PyTorch, it is a grouped convolution with as many groups as channels: `nn.Conv2d(64, 64, 3, padding=1, groups=64)` has 64 filters of 1 × 3 × 3. On its own, it mixes nothing: what is in channel 5 stays in channel 5.

        **The depthwise separable convolution.** A 1×1 convolution after it does the second job: it mixes the channels of each pixel. The paper calls it the pointwise convolution, because it looks at one pixel at a time. Together they are a **depthwise separable convolution**: first filter each channel, then mix. For one pixel of the output, with $M$ channels in, $N$ out and filters of $k$ by $k$, an ordinary convolution needs $k^2 M N$ multiplications; the depthwise convolution $k^2 M$, and the 1×1 convolution $M N$. Divided by the ordinary convolution's, that is

        $$\frac{k^2 M + M N}{k^2 M N} = \frac{1}{N} + \frac{1}{k^2}.$$

        With 3×3 filters and many channels, that is a little more than 1/9: 8 to 9 times fewer multiplications, as the paper says. With stride 1, a convolution uses each of its weights once for every pixel it gives, so the weights shrink just as much, biases aside.

        The picture: the weights of three convolutions with 4 channels in, a column for each channel and a row for each filter. A 3×3 convolution to 3 channels has 3×3 weights in every place; the depthwise convolution only on the diagonal, because each filter sees only its own channel; the 1×1 convolution to 3 channels one weight in every place.
        """),
        C.filters_picture({"full": "3×3 convolution", "depthwise": "depthwise convolution", "pointwise": "then 1×1 convolution",
                           "channels": "{n} channels in", "filters": "filters", "weights": "{n} weights",
                           "alt": "Three grids of weights, each with 4 columns for the 4 channels that come in. The 3×3 convolution: 3 rows of filters, a 3×3 grid of weights in every place, 108 weights. The depthwise convolution: 4 rows, a 3×3 grid only where the row and the column are the same channel, the other places empty, 36 weights. The 1×1 convolution: 3 rows with one weight in every place, 12 weights."}),
        mo.md(r"""
        **MobileNet** is one ordinary 3×3 convolution followed by 13 depthwise separable convolutions, with batch normalization and a ReLU after every depthwise and every 1×1 convolution. A few of its depthwise convolutions have stride 2, which makes the maps smaller. The paper's table 3 counts one layer of it, with 512 channels in and out on maps of 14 by 14:

        | One layer | Multiplications | Weights |
        |---|---|---|
        | 3×3 convolution | 462 million | 2.36 million |
        | depthwise separable convolution | 52.3 million | 0.27 million |

        And whole networks on ImageNet, from its tables 4 and 8:

        | Network | Right on ImageNet | Multiplications for a photo | Parameters |
        |---|---|---|---|
        | MobileNet with ordinary 3×3 convolutions | 71.7% | 4,866 million | 29.3 million |
        | MobileNet | 70.6% | 569 million | 4.2 million |
        | GoogLeNet | 69.8% | 1,550 million | 6.8 million |
        | VGG-16 | 71.5% | 15,300 million | 138 million |
        """),
        L.watch_out("Fewer multiplications do not always make a layer that much faster. How fast a layer runs also depends on how much work it does for each number it reads: a depthwise convolution uses each number in 9 multiplications, a 3×3 convolution from 64 channels to 64 in 576. The paper put 95% of MobileNet's multiplications in its 1×1 convolutions, which a computer can do as one large matrix multiplication, a job it is fast at. And the paper's numbers come from ImageNet, with batch normalization and photos of 224 by 224: what the split does in a small network on CIFAR-10, the demo has to show."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **depthwise convolution** | a convolution that gives every channel a filter of its own, which sees only that channel: a grouped convolution with as many groups as channels | `nn.Conv2d(64, 64, 3, padding=1, groups=64)` |
        | **depthwise separable convolution** | a depthwise convolution with a 1×1 convolution after it: first filter each channel, then mix the channels | `SeparableConv(64, 64)` |
        | **MobileNet** | a CNN for phones built from depthwise separable convolutions; from 2017 | — |
        | $M$, $N$ | the channels that come into a convolution, and the channels that come out | `channels_in`, `channels_out` |

        **The demo's network.** The demo trains the network of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), with a depthwise separable convolution from 64 channels to 64 in its middle, on the 64 maps of 8 by 8: 4,800 parameters and 299,008 multiplications for a photo, where one 3×3 convolution, as in that lesson, has 36,928 and 2,359,296. To see what each step does, its middle can also be two depthwise separable convolutions in a row, with 9,600 parameters, only the depthwise convolution, with 640, or only the 1×1 convolution, with 4,160.

        The picture: how many of the first 1,000 test photos of CIFAR-10 the network reads right after 20 epochs on the first 6,000 training photos, a dot for each of 5 seeds, with the course's code. The row at the top is from that lesson: one 3×3 convolution, 532 to 572. The depthwise separable convolution: 548 to 581. Two in a row: 527 to 541. Only the depthwise convolution: 557 to 585; only the 1×1 convolution: 557 to 576. Exercise 3 asks what the rows say.
        """),
        mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                "plain": "one 3×3 convolution", "separable": "depthwise separable convolution",
                                "two": "two depthwise separable convolutions", "depthwise": "only the depthwise convolution",
                                "pointwise": "only the 1×1 convolution", "mine": "your run", "course": "the course's runs",
                                "alt": "Five rows of open dots, test photos read right of 1,000, one dot for each seed and a short line at their mean. One 3×3 convolution: 532 to 572. Depthwise separable convolution: 548 to 581. Two of them in a row: 527 to 541. Only the depthwise convolution: 557 to 585. Only the 1×1 convolution: 557 to 576."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a depthwise separable convolution from `channels_in` channels to `channels_out`:

    1. In `__init__`, make the depthwise convolution: a 3×3 convolution from `channels_in` channels to `channels_in`, with `padding=1` and `groups=channels_in`, so that every channel gets a filter of its own.
    2. Make the 1×1 convolution, from `channels_in` channels to `channels_out`.
    3. In `forward`, send `x` through the depthwise convolution and a ReLU, then through the 1×1 convolution and a ReLU.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes a 3×3 convolution from 64 channels to 64, and the two steps of a depthwise separable convolution that do its work, and sends made-up maps of one photo, 64 channels of 8 by 8, through both. It prints the shapes of what comes out, the shapes of the weights, and the multiplications of each for one photo: its weights times the 8 × 8 pixels it gives. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_separable():
    import torch
    from torch import nn

    x = torch.rand(1, 64, 8, 8)
    channels_out = 64  # change: channels_out = 512
    full = nn.Conv2d(64, channels_out, 3, padding=1)
    # one 3×3 filter for each of the 64 channels, then a 1×1 convolution that mixes them
    depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
    pointwise = nn.Conv2d(64, channels_out, 1)
    with torch.no_grad():
        print(list(full(x).shape), list(pointwise(depthwise(x)).shape))
    print(list(full.weight.shape), list(depthwise.weight.shape), list(pointwise.weight.shape))
    # every weight is used once for each of the 8 × 8 pixels of the output
    full_mults = full.weight.numel() * 8 * 8
    separable_mults = (depthwise.weight.numel() + pointwise.weight.numel()) * 8 * 8
    print(full_mults, separable_mults, full_mults / separable_mults)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `[1, 64, 8, 8] [1, 64, 8, 8]`: both give 64 maps of 8 by 8. Then `[64, 64, 3, 3] [64, 1, 3, 3] [64, 64, 1, 1]`: the 3×3 convolution has 64 filters of 64 × 3 × 3, the depthwise convolution 64 filters of 1 × 3 × 3, one for each channel, and the 1×1 convolution 64 filters of 64 × 1 × 1. And `2359296 299008 7.890410958904109`: 7.89 times fewer multiplications. With `channels_out = 512`, the depthwise convolution stays the same, the other two get 512 filters, and the cell prints `18874368 2134016 8.844529750479847`: 8.84 times fewer, as in the paper's layer with 512 channels.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide the channels that come out, from 4 to 2,048, with as many coming in. The top bar is the 3×3 convolution's multiplications, 100%. Under it, the depthwise separable convolution's: light, the 1×1 convolution's; dark, the depthwise convolution's after it. The dashed line is at 1/9.
    """)
    return


@app.cell(hide_code=True)
def play_channels():
    _labels = {"title": "A depthwise separable convolution against a 3×3 one", "x": "multiplications, in % of the 3×3 convolution's",
               "full": "3×3 convolution: 100%", "separable": "depthwise separable: {t}%", "floor": "the 1×1 alone: 1/9",
               "alt": "Two bars of multiplications, in % of the 3×3 convolution's: the 3×3 convolution at 100%, and under it the depthwise separable convolution, the 1×1 convolution's 1/9 light and the depthwise convolution's share dark after it, with a dashed line at 1/9."}
    explore_channels = L.explore("channels-slider", "channels", lambda _v: (C.share_picture(int(_v), _labels, L.lang), C.share_sentence(int(_v), L.lang)),
                                 start=64, values=C.CHANNELS_OUT, role="hyper",
                                 task="Slide the channels up to 2,048. How many times fewer multiplications does the depthwise separable convolution get close to, without ever reaching it?")
    explore_channels
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Close to 9, never 9: the 1×1 convolution alone needs 1/9 of the 3×3 convolution's multiplications, however many channels there are, because a 1×1 filter covers 1 pixel where a 3×3 filter covers 9. The depthwise convolution adds $1/N$ on top, which shrinks as the channels grow. With 4 channels, the two together save only 2.77 times; with 64, as in the demo, 7.89; with 512, as in the paper's layer, 8.84; with 2,048, 8.96. With few channels, the depthwise convolution is a large part of the cost; with many, nearly all of it is in the 1×1 convolution.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which step does what

    Six sentences about the two steps of a depthwise separable convolution with 3×3 filters. Which describe the depthwise convolution, and which the 1×1 convolution after it? Sort each one.
    """)
    return


@app.cell(hide_code=True)
def exercise_steps():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps",
        [
            "The depthwise convolution has 3×3 filters that each see one channel; the 1×1 convolution has filters of 1 by 1 that see all the channels. Which of the two can see neighbors, and which can mix?",
            "Three sentences belong to each. For the multiplications of one pixel: 9 × 64 for the depthwise convolution, 64 × 64 for the 1×1 convolution.",
        ],
        why="**The depthwise convolution:** it looks at the 3×3 neighbors of every pixel, each of its filters sees only one channel, and its `groups` is the number of channels it gets. **The 1×1 convolution:** it mixes the channels of every pixel, it decides how many channels come out, and with 64 channels in and out it does most of the multiplications, 4,096 for a pixel against 576.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_separable():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A depthwise separable convolution

        Complete `SeparableConv`. Its arguments are the channels that come in, `channels_in`, and the channels that go out, `channels_out`. `forward` is in the box: it sends `x` through `self.depthwise` and a ReLU, then through `self.pointwise` and a ReLU. Make the two convolutions: the depthwise one, a 3×3 filter for each channel on its own, that keeps the height and width, and the 1×1 one. The check builds your block, with 64 channels in and out and with other numbers, and compares what comes out, the shapes of its weights and how many parameters it has with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/separable.py`.
        """),
        L.task("Make the depthwise convolution and the 1×1 convolution."),
    ])
    return


@app.cell(hide_code=True)
def exercise_separable():
    box_separable = L.code_box("separable", lines=16)
    box_separable
    return (box_separable,)


@app.cell(hide_code=True)
def check_separable(box_separable):
    result_separable = L.check("separable", box_separable.value)
    result_separable
    return (result_separable,)


@app.cell(hide_code=True)
def hints_separable():
    L.hints(
        "separable",
        [
            "The depthwise convolution makes one map from each channel it gets: how many filters does it have, and how many groups? A 3×3 filter keeps the height and width with `padding=1`.",
            "`self.depthwise = nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in)`. The 1×1 convolution is step 2 of the recipe.",
        ],
        why="The depthwise convolution is `nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in)`, and the 1×1 convolution `nn.Conv2d(channels_in, channels_out, 1)`. MobileNet puts batch normalization after each of the two, before its ReLU, and gives a few of its depthwise convolutions stride 2.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains the demo's network with the middle you choose on the first 6,000 training photos of CIFAR-10, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Choose the middle: your `SeparableConv`, two of them in a row, or, from the course, only the depthwise convolution or only the 1×1 convolution, which need no block of yours. Where a module of yours from an earlier lesson is missing or broken, the course's stands in. It needs CIFAR-10, and for your own block exercise 2, and takes a few minutes. While it trains, a chart shows the loss on the training photos every 10 batches, and dashed on the test photos after every epoch. Beside or under it are the network's guesses at that moment for one test photo of each kind, and under them the feature maps its middle makes of a car; both are drawn again as it trains. In the picture after, the gray line is the validation loss of the course's run with one 3×3 convolution in the middle and seed 0. Then you can rewind to a few epochs of your run, with both pictures as they were then, and see your run next to the course's.
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
    variant_choice = mo.ui.radio(options={"your depthwise separable convolution": "separable", "two of them in a row": "two",
                                          "only the depthwise convolution": "depthwise", "only the 1×1 convolution": "pointwise"},
                                 value="your depthwise separable convolution", label="The middle of the network")
    train_button = mo.ui.run_button(label="Train the network")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_separable(cifar, train_button, variant_choice, result_separable):
    cifar, result_separable  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the network.*")
    else:
        _variant = variant_choice.value
        _names = {"separable": "Depthwise separable", "two": "Two depthwise separable", "depthwise": "Only depthwise", "pointwise": "Only 1×1"}
        _with = {"separable": "With your depthwise separable convolution", "two": "With two of your depthwise separable convolutions in a row",
                 "depthwise": "With only the depthwise convolution", "pointwise": "With only the 1×1 convolution"}
        _now = {}   # the live chart, made at the first report
        _layer = "The middle of the network on a test photo of a car: layer {name}"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your network on CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "The loss of the network so far: on the training photos every 10 batches, and dashed on the 1,000 test photos after every epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "training photos, every 10 batches"), ("loss", True, "test photos, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-mobilenet-now"))
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
            _out = mo.md("*Finish exercise 2 first: then your `SeparableConv` is in your project, and this cell uses it. Only the depthwise or only the 1×1 convolution needs no block of yours.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `SeparableConv` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "separable": "*Your `SeparableConv` in `work/projects/cnn/separable.py` stops with an error, or does not give back what it should: with 64 channels in and out, 64 maps of 8 by 8. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "cnn": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give what it should. Did you change its file? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "one 3×3 convolution",
                       "alt": "The loss of the network after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's run with one 3×3 convolution in the middle and seed 0, which falls to epoch 6 and then rises."}
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
                mo.md(f"""{_with[_variant]}, the network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters, {C.grouped(_result['middle_params'], L.lang)} of them in its middle.{_turn} Over 5 seeds, with the course's code, this middle: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos, with the lowest validation loss after epoch {_be[0]} to {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                        "plain": "one 3×3 convolution", "separable": "depthwise separable convolution",
                                        "two": "two depthwise separable convolutions", "depthwise": "only the depthwise convolution",
                                        "pointwise": "only the 1×1 convolution", "mine": "your run", "course": "the course's runs",
                                        "alt": "Five rows of test photos read right, of 1,000: open dots for the course's runs over 5 seeds, with one 3×3 convolution, with the depthwise separable convolution, with two of them, and with only the depthwise or only the 1×1 convolution, with a short line at the mean of each row; and in the row of your middle a filled dot for your run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Exercise 3. What the split does here
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "separable-runs",
        question="Over 5 seeds, after 20 epochs. With one 3×3 convolution in its middle, the network reads 532 to 572 test photos right; with a depthwise separable convolution, 548 to 581, with 8 times fewer multiplications and weights in its middle. With only the depthwise convolution it reads 557 to 585, with only the 1×1 convolution 557 to 576, and with two depthwise separable convolutions in a row 527 to 541. What do the numbers say?",
        options={
            "both": "Each step needs the other: alone, the depthwise or the 1×1 convolution reads fewer photos right than the two together",
            "more": "The more depthwise separable convolutions, the better: two in a row read the most",
            "right": "Here the split costs nothing: the depthwise separable convolution keeps up with the 3×3 convolution at an eighth of the cost, but each step alone does as well, so these runs cannot show what each step adds",
        },
        correct="right",
        why={
            "both": "Alone, they read 557 to 585 and 557 to 576 right: as many as the two together, 548 to 581, or more. In this network, the middle may have little left to do: the two 3×3 convolutions before it already look at the neighbors and mix the channels, and the linear layer after it mixes everything. That each step is needed, these runs cannot show.",
            "more": "Two in a row read the fewest, 527 to 541, and learn fewer of their 6,000 training photos, 5,436 to 5,735, against 5,829 to 5,982 for one. More layers are not better here. MobileNet has batch normalization after every convolution; this network has none.",
            "right": "The depthwise separable convolution reads 548 to 581 test photos right, the 3×3 convolution 532 to 572: their runs overlap, with 299,008 multiplications in the middle for a photo against 2,359,296, and 4,800 parameters against 36,928. On ImageNet, in the paper, the split cost a little: 70.6% right against 71.7%; here, nothing you can see. But only the depthwise convolution, with 640 parameters, reads 557 to 585, and only the 1×1 convolution 557 to 576: in this small network, with two 3×3 convolutions before the middle, each step alone does as well. What each step adds, these runs cannot show: a finding holds where it was measured. And two in a row read fewer, 527 to 541.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A depthwise convolution is the grouped convolution of [Groups of channels: ResNeXt](/course/open/cv-resnext?lang=en) with as many groups as channels, and the 1×1 convolution after it is the one of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en). You counted their weights and multiplications with the shapes of [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), and trained them with the Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) on the photos of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en). Your project `cnn` now has `separable.py`. In the next lesson, *EfficientNet: wider, deeper or sharper*, a network built from such blocks grows in depth, in width and in the size of its photos, and you work out what each choice costs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_steps, result_separable, choice_runs):
    L.wrap_up(result_steps, result_separable, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
