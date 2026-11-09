# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-inception. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-inception", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en) and [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reduce():
    quick_reduce = L.choice(
        "quick-reduce",
        question="A 1×1 convolution from 192 channels to 16. How many parameters does it have?",
        options={
            "five": "76,816: 16 filters of 192 × 5 × 5 weights, each with a bias",
            "right": "3,088: 16 filters of 192 weights, each with a bias",
            "sum": "208: 192 + 16",
        },
        correct="right",
        why={
            "five": "That would be a convolution of 5 by 5. A filter of 1 by 1 has one weight for each channel that comes in, 192, and a bias. See [1×1 convolutions](/course/open/cv-1x1-convolution?lang=en).",
            "sum": "Each of the 16 filters has a weight for every one of the 192 channels, and a bias: 16 × (192 + 1). See [1×1 convolutions](/course/open/cv-1x1-convolution?lang=en).",
            "right": "16 × (192 + 1) = 3,088. Today such a convolution stands before a convolution of 5 by 5, which then works on 16 channels instead of 192.",
        },
    )
    quick_reduce
    return (quick_reduce,)


@app.cell(hide_code=True)
def quick_check_view():
    quick_view = L.choice(
        "quick-view",
        question="Two 3×3 convolutions with padding 1, one after the other. How wide a piece of the input does one number after them depend on?",
        options={
            "six": "6 by 6: 3 + 3",
            "right": "5 by 5",
            "nine": "9 by 9: 3 × 3",
        },
        correct="right",
        why={
            "six": "The two pieces overlap: every 3×3 convolution adds 2 to the width, not 3. See [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en).",
            "nine": "Every 3×3 convolution adds 2 to the width of the piece: 3 after the first, 5 after the second. See [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en).",
            "right": "3 by 3 after the first, 5 by 5 after the second, as one filter of 5 by 5 would see. VGG stacks filters of one size; today's block puts filters of several sizes side by side.",
        },
    )
    quick_view
    return (quick_view,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        How large should a filter be? In [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en), every filter was 3 by 3, and only a stack of them let a number see further. But what a photo shows comes in every size: the eye of a cat is a few pixels, its head can fill half the photo. In 2014, Christian Szegedy and his colleagues at Google gave one block filters of 1 by 1, 3 by 3 and 5 by 5 side by side, with max pooling beside them, and let training decide what each is used for. With the 1×1 convolutions of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en) in front, to keep that affordable, they stacked 9 such blocks in GoogLeNet: 22 layers with weights, and 12 times fewer parameters than AlexNet, the winner of 2012. It won ImageNet's contest of 2014. The name comes from *Network In Network* and from an internet joke about the film *Inception*: "we need to go deeper". Today you count what the 1×1 convolutions save, build an Inception block, and find out in a CNN on CIFAR-10 which of its branches the network leans on.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_channels():
    guess_channels = L.guess(
        "naive-channels",
        question="Inception (3a), a block of GoogLeNet, gets 192 channels of 28 by 28. Without the 1×1 convolutions that make the channels fewer, it would put three convolutions straight on those 192 channels, with 64 filters of 1 by 1, 128 of 3 by 3 and 32 of 5 by 5, and 3×3 max pooling beside them, and stack all that comes out along the channels. How many channels would come out?",
        options={
            "sum": "224: 64 + 128 + 32",
            "right": "416",
            "same": "192: as many as went in",
        },
        reveal="**416.** The pooling has no filters: it keeps all 192 channels, so 64 + 128 + 32 + 192 = 416 come out. The next block would get 416 channels, and with pooling of its own give even more: the paper calls that a computational blow up within a few stages. GoogLeNet puts a 1×1 convolution after the pooling, to 32 channels, so the block gives 256. And it puts 1×1 convolutions before the 3×3 and the 5×5 filters, which then work on 96 and 16 channels instead of 192: 163,696 parameters instead of 387,296.",
    )
    guess_channels
    return (guess_channels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Branches side by side.** An Inception block sends the same maps into four branches at once: a 1×1 convolution; a 1×1 convolution to fewer channels, then a 3×3 convolution; a 1×1 convolution to fewer channels, then a 5×5 convolution; and 3×3 max pooling, then a 1×1 convolution. Every convolution has a ReLU after it. Then the block puts what the four branches give side by side along the channels, with `torch.cat` of [PyTorch tensors](/course/open/torch-tensors?lang=en). A batch of maps has the shape (photos, channels, rows, columns), so the channels are `dim=1`, and they add up: in inception (3a), 64 + 128 + 32 + 32 = 256.

        **The same height and width.** `torch.cat` only puts maps side by side when their rows and columns match. With stride 1, a window of $k$ by $k$ fits $n + 2p - k + 1$ times across $n$ pixels with padding $p$, by the formula of [Convolution in detail](/course/open/cnn-conv-formal?lang=en): padding 1 keeps the size for 3 by 3, and padding 2 for 5 by 5. The max pooling of the fourth branch uses stride 1 and padding 1, so it keeps the size too, unlike the 2×2 pooling of your ColorCNN, which halves it.

        The picture: inception (3a), from the 192 maps of 28 by 28 that come in, through the four branches, to the 256 that come out. The orange boxes have weights that training learns; the pooling has none.
        """),
        C.block_picture(C.G3A, C.G3A_SIDE, {"title": "Inception (3a) of GoogLeNet", "input": "{c} channels of {s} by {s}",
                                            "output": "{c} channels of {s} by {s}, side by side", "filters": "{n} filters",
                                            "conv1": "1×1 convolution", "conv": "{k}×{k} convolution", "pool": "3×3 max pooling",
                                            "alt": "An Inception block from top to bottom. 192 channels of 28 by 28 go into four branches side by side: a 1×1 convolution with 64 filters; a 1×1 convolution with 96 filters, then a 3×3 convolution with 128; a 1×1 convolution with 16 filters, then a 5×5 convolution with 32; and 3×3 max pooling, then a 1×1 convolution with 32 filters. At the bottom, 256 channels of 28 by 28: 64 + 128 + 32 + 32."}),
        mo.md(r"""
        **Why the 1×1 convolutions.** A filter of $k$ by $k$ on $c$ channels has $c \times k \times k$ weights, so a 5×5 filter on all 192 channels is expensive: 4,800 weights. Behind a 1×1 convolution to 16 channels it has 400. Inception (3a) has 163,696 parameters; GoogLeNet's own table says 159K, because it counts only the 163,328 weights, without the biases, and counts a K as 1,024. For one photo the block does 128,049,152 multiplications: those 163,328 for each of the 28 × 28 pixels, the 128 million in the table. Without its 1×1 convolutions before the 3×3 and 5×5 filters and after the pooling, the naive block of the paper, it would have 387,296 parameters, do 303 million multiplications, and give the next block 416 channels instead of 256.
        """),
        L.watch_out("Inception is the name of a family, not of one network. GoogLeNet is the first, from 2014; the versions after it, from 2015 on, changed the block. Some put two 3×3 convolutions where the 5×5 was, the trick of VGG blocks, and added batch normalization. When a paper or a library says Inception, check which version it means."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **Inception block** | branches of 1×1, 3×3 and 5×5 convolutions and of 3×3 max pooling side by side on the same maps, the 3×3 and 5×5 behind a 1×1 convolution to fewer channels and the pooling before one, their maps stacked along the channels. GoogLeNet is a stack of them | `torch.cat([y1, y3, y5, yp], dim=1)` |

        **The demo's network.** The demo trains the network of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), with an Inception block in its middle. On the 64 maps of 8 by 8, it has 16 filters of 1 by 1; 32 of 3 by 3, behind a 1×1 convolution to 32 channels; 8 of 5 by 5, behind one to 8; and 8 filters of 1 by 1 after the pooling: 16 + 32 + 8 + 8 = 64 channels, the outputs of inception (3a) divided by 4. The block has 15,016 parameters, against 36,928 for one 3×3 convolution from 64 to 64; the whole network 44,658 against 66,570.

        The picture: how many of the first 1,000 test photos of CIFAR-10 the network reads right after 20 epochs on the first 6,000 training photos, a dot for each of 5 seeds, with the course's code. With one 3×3 convolution in the middle, as in that lesson: 532 to 572, 549 on average. With the Inception block: 553 to 585, 567 on average. The ranges overlap: at least as many, with a third fewer parameters. It still learns its training photos almost by heart, 5,896 to 5,997 of 6,000. Then the course switched off one branch at a time in the trained networks, by setting its channels to 0, and counted again: the four rows below. Exercise 3 asks what they say.
        """),
        mo.Html(C.off_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                               "plain": "one 3×3 convolution", "inception": "Inception block, all four branches",
                               "b1": "without its 1×1 branch", "b3": "without its 3×3 branch", "b5": "without its 5×5 branch",
                               "pool": "without its pooling branch", "mine": "your run", "course": "the course's runs",
                               "alt": "Six rows of open dots, test photos read right of 1,000, between 400 and 600, one dot for each seed and a short line at their mean. One 3×3 convolution in the middle: 532 to 572. The Inception block with all four branches: 553 to 585. Without its 1×1 branch: 409 to 484. Without its 3×3 branch: 453 to 504. Without its 5×5 branch: 521 to 567. Without its pooling branch: 468 to 541."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build an Inception block:

    1. Give it four branches on the same input `x`: a 1×1 convolution; a 1×1 convolution to fewer channels and a 3×3 convolution; a 1×1 convolution to fewer channels and a 5×5 convolution; 3×3 max pooling and a 1×1 convolution.
    2. Put a ReLU after every convolution.
    3. Keep the height and width in every branch: padding 1 for 3 by 3, padding 2 for 5 by 5, and the max pooling with `stride=1, padding=1`.
    4. Put the branches side by side along the channels, `torch.cat([y1, y3, y5, yp], dim=1)`: then `c1 + c3 + c5 + pool_proj` channels come out.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell builds inception (3a) from its six convolutions and its pooling, sends made-up maps of one photo through it, 192 channels of 28 by 28, and prints the height and width of what each branch gives, the shape of the four side by side, and the number of parameters. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_inception():
    import torch
    from torch import nn

    x = torch.rand(1, 192, 28, 28)
    # 1. four branches on the same x
    b1 = nn.Conv2d(192, 64, 1)
    b3_reduce, b3 = nn.Conv2d(192, 96, 1), nn.Conv2d(96, 128, 3, padding=1)
    b5_reduce, b5 = nn.Conv2d(192, 16, 1), nn.Conv2d(16, 32, 5, padding=2)  # change: padding=1
    pool, pool_proj = nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(192, 32, 1)
    # 2. a ReLU after every convolution
    y1 = torch.relu(b1(x))
    y3 = torch.relu(b3(torch.relu(b3_reduce(x))))
    y5 = torch.relu(b5(torch.relu(b5_reduce(x))))
    yp = torch.relu(pool_proj(pool(x)))
    # 3. every branch keeps 28 by 28
    print(y1.shape[2:], y3.shape[2:], y5.shape[2:], yp.shape[2:])
    # 4. side by side along the channels
    out = torch.cat([y1, y3, y5, yp], dim=1)
    print(out.shape)
    total = 0
    for layer in (b1, b3_reduce, b3, b5_reduce, b5, pool_proj):
        total += sum(q.numel() for q in layer.parameters())
    print(total)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `torch.Size([28, 28])` four times: every branch keeps the height and width, as step 3 of the recipe says. Then `torch.Size([1, 256, 28, 28])`: 64 + 128 + 32 + 32 channels side by side, and `163696` parameters. 12,352 of them are in the 1×1 branch; 18,528 + 110,720 in the 3×3 branch; 3,088 + 12,832 in the 5×5 branch; and 6,176 in the 1×1 convolution after the pooling. With `padding=1`, the 5×5 convolution gives 26 by 26: the first line prints `torch.Size([26, 26])` for the third branch, and `torch.cat` stops with `Sizes of tensors must match except in dimension 1. Expected size 28 but got size 26 for tensor number 2 in the list`. It counts the tensors in the list from 0, so number 2 is `y5`.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide the size of the filters of one branch, from 1 by 1 to 11 by 11. The branch gets 192 channels and gives 32, as the 5×5 branch of inception (3a). The gray bar is the branch straight on the 192 channels; the orange bar is the branch behind a 1×1 convolution to 16 channels, its two convolutions one after the other. The axis stays the same, in thousands of parameters.
    """)
    return


@app.cell(hide_code=True)
def play_kernel():
    _labels = {"title": "One branch, from 192 channels to 32", "x": "parameters (thousands)",
               "straight": "straight on 192", "reduced": "behind a 1×1 to 16",
               "alt": "Two bars of parameters, in thousands, for filters of the chosen size: the branch straight on the 192 channels, in gray, and the branch behind a 1×1 convolution to 16 channels, its two convolutions one after the other, in orange."}
    explore_kernel = L.explore("kernel", "filter size", lambda _v: (C.kernel_picture(int(_v), _labels, L.lang), C.kernel_sentence(int(_v), L.lang)),
                               start=5, values=C.KERNELS, role="hyper",
                               task="From which filter size on does the branch straight on the 192 channels have more than 10 times as many parameters as the branch behind the 1×1 convolution?")
    explore_kernel
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    From 7 by 7: 301,088 parameters against 28,208, 10.7 times as many. With 5 by 5 it is 9.7 times, and with 3 by 3, 7.2. The larger the filter, the closer the factor comes to 192 / 16 = 12: then nearly all parameters are in the filters of $k$ by $k$, and behind the 1×1 convolution those work on 16 channels instead of 192. That is why GoogLeNet gives its 5×5 branches the narrowest 1×1 convolutions in front: 16 of 192 channels in inception (3a), where the 3×3 branch gets 96.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_same_size():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The same size

    A branch of an Inception block gets maps of 28 by 28, and `torch.cat` needs the same height and width from all four. Which of these six steps keep 28 by 28, and which change the size? Put each in its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_same_size():
    sort_same_size = L.sort("same-size")
    sort_same_size
    return (sort_same_size,)


@app.cell(hide_code=True)
def check_same_size(sort_same_size):
    result_same_size = L.check("same-size", sort_same_size)
    result_same_size
    return (result_same_size,)


@app.cell(hide_code=True)
def hints_same_size():
    L.hints(
        "same-size",
        [
            "For each step, work out how many places a window of $k$ by $k$ has across 28 pixels: $28 + 2p - k + 1$ with stride 1. With a stride of 2, it takes every second place only.",
            "Four of them keep 28 by 28, two do not. A padding of $(k - 1) / 2$ keeps the size: 0 for 1 by 1, 1 for 3 by 3, 2 for 5 by 5.",
        ],
        why="**They keep 28 by 28:** the 1×1 convolution, 28 + 0 − 1 + 1 = 28; the 3×3 convolution with padding 1, 28 + 2 − 3 + 1 = 28; the 5×5 convolution with padding 2, 28 + 4 − 5 + 1 = 28; and 3×3 max pooling with stride 1 and padding 1, as the 3×3 convolution. **They change the size:** the 5×5 convolution with padding 1, 28 + 2 − 5 + 1 = 26; and 2×2 max pooling, whose stride is 2, so 14 by 14.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_inception():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. An Inception block

        Complete `Inception`. Its arguments are the columns of GoogLeNet's table: the channels that come in, `c_in`; the filters of the 1×1 branch, `c1`; the filters of the 1×1 convolution in front of the 3×3 one, and of the 3×3 one itself, `c3_reduce` and `c3`; the same for the 5×5 branch, `c5_reduce` and `c5`; and the filters after the pooling, `pool_proj`. The 1×1, 3×3 and pooling branches are in the box. Make the 5×5 branch: `b5_reduce`, a 1×1 convolution from `c_in` to `c5_reduce`, and `b5`, a 5×5 convolution from `c5_reduce` to `c5` that keeps the height and width. In `forward`, make `y5` as `y3` is made, and give back the four branches side by side along the channels, in the order `y1`, `y3`, `y5`, `yp`. The check builds your block, with the numbers of inception (3a) and with made-up ones, and compares what comes out, and how many parameters it has, with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/inception.py`.
        """),
        L.task("Make `b5_reduce` and `b5`, then `y5`, and give back the four branches side by side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_inception():
    box_inception = L.code_box("inception", lines=27)
    box_inception
    return (box_inception,)


@app.cell(hide_code=True)
def check_inception(box_inception):
    result_inception = L.check("inception", box_inception.value)
    result_inception
    return (result_inception,)


@app.cell(hide_code=True)
def hints_inception():
    L.hints(
        "inception",
        [
            "`b5_reduce` and `b5` are made as `b3_reduce` and `b3` are, with other names and another kernel size. Which padding keeps the height and width for 5 by 5? Exercise 1 has it.",
            "`self.b5 = nn.Conv2d(c5_reduce, c5, 5, padding=2)`, and `y5` is the line of `y3` with `b5_reduce` and `b5`. The last line is step 4 of the recipe.",
        ],
        why="`b5_reduce` is `nn.Conv2d(c_in, c5_reduce, 1)` and `b5` is `nn.Conv2d(c5_reduce, c5, 5, padding=2)`; `y5 = torch.relu(self.b5(torch.relu(self.b5_reduce(x))))`; and `forward` ends with `return torch.cat([y1, y3, y5, yp], dim=1)`. GoogLeNet's own blocks are like this, without batch normalization, which came the next year.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains the demo's network, with your `Inception` in its middle, on the first 6,000 training photos of CIFAR-10, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Then it switches off each branch of your block in turn, by setting its channels to 0, and counts the test photos again. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10 and exercise 2, and takes a few minutes. While it trains, a chart shows the validation loss after every epoch. In the picture after, the gray line is the validation loss of the course's run with one 3×3 convolution in the middle and seed 0. Then you can rewind through your run, and see your run next to the course's.
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
def train_button_cell():
    train_button = mo.ui.run_button(label="Train the network")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_inception(cifar, train_button, result_inception):
    cifar, result_inception  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the network.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your network on CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "The validation loss of the network on the 1,000 test photos after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Inception block** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Inception block** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `Inception` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `Inception` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "inception": "*Your `Inception` in `work/projects/cnn/inception.py` stops with an error, or does not give back 64 maps of 8 by 8. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "cnn": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give what it should. Did you change its file? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "one 3×3 convolution",
                       "alt": "The loss of the network with the Inception block after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's run with one 3×3 convolution in the middle and seed 0, which falls to epoch 6 and then rises."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.guesses_picture(_guesses[_e - 1], _result["truth"], L.lang),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.SHOWN)} photos read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: when is the validation loss lowest, and what happens after?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Its validation loss was lowest after epoch **{_best}**." if _best < C.EPOCHS else f" Its validation loss was still falling after epoch {_best}.")
            _ref = C.REFERENCE["inception"]
            _tr, _te = C.low_high(_ref["train"]), C.low_high(_ref["test"])
            _off = _result["off"]
            _out = mo.vstack(_pic + [
                mo.md(f"""With your Inception block, the network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters, {C.grouped(_result['block_params'], L.lang)} of them in its block.{_turn} With one branch switched off, it reads {_off['b1']} test photos right without its 1×1 branch, {_off['b3']} without its 3×3 branch, {_off['b5']} without its 5×5 branch, and {_off['pool']} without its pooling branch. Over 5 seeds, with the course's code: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos.{_who}"""),
                mo.Html(C.off_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                       "plain": "one 3×3 convolution", "inception": "Inception block, all four branches",
                                       "b1": "without its 1×1 branch", "b3": "without its 3×3 branch", "b5": "without its 5×5 branch",
                                       "pool": "without its pooling branch", "mine": "your run", "course": "the course's runs",
                                       "alt": "Six rows of test photos read right, of 1,000: open dots for the course's runs over 5 seeds, with one 3×3 convolution in the middle, with the Inception block, and with each of its four branches switched off, with a short line at the mean of each row; and in the rows of the Inception block a filled dot for your run."}, L.lang, mine=_result)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_branches():
    mo.md(r"""
    ### Exercise 3. Which branch the network leans on
    """)
    return


@app.cell(hide_code=True)
def choice_branches_cell():
    choice_branches = L.choice(
        "branches",
        question="Over 5 seeds, after 20 epochs, the network with the Inception block reads 553 to 585 test photos right. With one branch switched off: without its 1×1 branch, 16 channels, 409 to 484; without its 3×3 branch, 32 channels, 453 to 504; without its 5×5 branch, 8 channels, 521 to 567; without its pooling branch, 8 channels, 468 to 541. What do the numbers say?",
        options={
            "large": "The 5×5 branch matters most: its filters see the most of each photo",
            "wide": "The more channels a branch gives, the more the network needs it: the 3×3 branch matters most",
            "right": "The network leans on its 1×1, 3×3 and pooling branches, and least on its 5×5 branch",
        },
        correct="right",
        why={
            "large": "Without its 5×5 branch, the network reads 521 to 567 right: from 2 more to 34 fewer than with all four, the smallest change of the four. On maps of 8 by 8, a 5×5 filter sees most of the map, but the network leans on that branch the least.",
            "wide": "The 3×3 branch gives the most channels, 32, but without the 1×1 branch, with 16, the network loses more: 100 to 144 test photos, against 51 to 107. And the pooling branch, with only 8 channels, costs 44 to 87.",
            "right": "Without its 1×1 branch, it reads 100 to 144 fewer right; without its 3×3 branch, 51 to 107; without its pooling branch, 44 to 87; without its 5×5 branch, from 2 more to 34 fewer. Training decided what each branch does: in this small network, on maps of 8 by 8, it leans least on the 5×5 branch. Switching a branch off after training only shows what the network leans on: trained without it, it might have done as well. Later Inception versions put two 3×3 convolutions where the 5×5 was: they see as far, with 18 weights per channel instead of 25.",
        },
    )
    choice_branches
    return (choice_branches,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** An Inception block puts the 1×1 convolutions of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en) in front of filters of 3 by 3 and 5 by 5 side by side, where [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en) stacked filters of one size after each other. You kept the height and width with the formula of [Convolution in detail](/course/open/cnn-conv-formal?lang=en), stacked the branches with `torch.cat` of [PyTorch tensors](/course/open/torch-tensors?lang=en), and trained the block with the Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) on the photos of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en). Your project `cnn` now has `inception.py`. In the next lesson, *Groups of channels: ResNeXt*, a block gets many thin paths side by side again, all of the same shape this time, with a convolution that splits the channels into groups.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_same_size, result_inception, choice_branches):
    L.wrap_up(result_same_size, result_inception, choice_branches)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
