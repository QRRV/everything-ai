# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-resnext. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-resnext", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Inception: filters of several sizes side by side](/course/open/cv-inception?lang=en) and [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cat():
    quick_cat = L.choice(
        "quick-cat",
        question="The four branches of an Inception block give 16, 32, 8 and 8 maps of 8 by 8, for a batch of 2 photos. What shape does `torch.cat([y1, y3, y5, yp], dim=1)` give?",
        options={
            "right": "(2, 64, 8, 8)",
            "stack": "(2, 4, 16, 8, 8): four branches of 16",
            "batch": "(8, 16, 8, 8): the photos of the four branches one after the other",
        },
        correct="right",
        why={
            "right": "Along `dim=1`, the channels, the maps are put side by side: 16 + 32 + 8 + 8 = 64. Today a convolution does the opposite: it splits its channels into groups, and gives each group filters of its own.",
            "stack": "`torch.cat` makes no new axis: it puts the tensors together along one they already have, here the channels. And the branches give different numbers of channels, so they could not even be stacked. See [Inception](/course/open/cv-inception?lang=en).",
            "batch": "Along `dim=0`, the photos, `torch.cat` would stop: the branches give different numbers of channels. Along `dim=1`, the channels, the batch keeps its 2 photos, each with 16 + 32 + 8 + 8 maps. See [Inception](/course/open/cv-inception?lang=en).",
        },
    )
    quick_cat
    return (quick_cat,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="A residual block gives back $x + f(x)$. Its input $x$ is 64 channels of 8 by 8. What must $f(x)$ give?",
        options={
            "right": "64 channels of 8 by 8: the shape of $x$",
            "any": "Any number of channels: `+` puts them side by side",
            "fewer": "32 channels of 8 by 8: inside the block, the channels get fewer",
        },
        correct="right",
        why={
            "right": "In ResNet's block, $f(x)$ has the shape of $x$, so that `+` adds a number of its own to every number of $x$. Today's block makes the channels fewer inside $f$, and then more again at its end, so that it fits.",
            "any": "Putting maps side by side is `torch.cat`. `+` adds them number by number, and stops when the shapes do not fit. See [ResNet](/course/open/cnn-resnet?lang=en).",
            "fewer": "Then `+` stops: 32 channels do not fit on 64, number by number. A block may make the channels fewer inside $f$, as long as $f$ ends with 64 again. See [ResNet](/course/open/cnn-resnet?lang=en).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Inception: filters of several sizes side by side](/course/open/cv-inception?lang=en), every branch had a design of its own: filters of 1, 3 or 5 by 5, with numbers of filters chosen by hand for each of GoogLeNet's 9 blocks. In 2016, Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu and Kaiming He, one of the makers of [ResNet](/course/open/cnn-resnet?lang=en), kept the many paths side by side, but gave them all the same shape: 32 thin paths in one residual block. Because the paths are alike, the whole block is three convolutions, the middle one a grouped convolution: a trick as old as AlexNet, which split its filters over two GPUs. Their ResNeXt-50 has about as many parameters as ResNet-50, 25.0 million against 25.5, and its first guess was wrong for 22.2% of ImageNet's validation photos, against 23.9%. ResNeXt took second place in ImageNet's contest of 2016. Today you count what groups save, build a ResNeXt block, and test in a CNN on CIFAR-10 whether more groups help there too.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "grouped-weights",
        question="A 3×3 convolution from 128 channels to 128, `nn.Conv2d(128, 128, 3)`, has 147,456 weights: 128 filters, each of 128 × 3 × 3. With `groups=32`, the channels and the filters are split into 32 groups, and each filter sees only the channels of its own group. How many weights then?",
        options={
            "same": "147,456: the same 128 filters, only in groups",
            "right": "4,608",
            "one": "144: 4 filters of 4 × 3 × 3",
        },
        reveal="**4,608, 32 times fewer.** Each of the 32 groups gets 4 of the 128 channels and 4 of the 128 filters, so each filter is 4 × 3 × 3 = 36 weights deep instead of 128 × 3 × 3 = 1,152. 128 filters of 36: 4,608. One group alone has 144, and there are 32 of them. What comes out still has 128 channels of the same size: only which channels each filter sees has changed.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Groups.** In a normal convolution, every filter sees all the channels that come in. `nn.Conv2d(c_in, c_out, k, groups=G)` splits the channels that come in into $G$ groups of $c_{in} / G$, and its filters too: each group of $c_{out} / G$ filters sees only the channels of its own group. Its weights get the shape $(c_{out}, c_{in} / G, k, k)$, $G$ times fewer, and what comes out is the groups side by side along the channels, in their order. So a grouped convolution is $G$ small convolutions side by side, their maps put together as `torch.cat` does: the worked example shows it. Both $c_{in}$ and $c_{out}$ must be divisible by $G$.

        The picture: a convolution from 8 channels to 8, a row for each filter and a column for each channel that comes in. An orange square stands for the 3 × 3 weights with which that filter looks at that channel. With one group, every filter sees all 8 channels; with 4 groups, each filter sees the 2 of its own group: 16 squares instead of 64.
        """),
        C.groups_picture({"title": "Which channels each filter sees", "in": "channels in", "out": "filters",
                                  "sub": "groups={g}: {n} squares",
                                  "alt": "Two grids of 8 by 8 squares, a row for each filter and a column for each channel that comes in. Left, groups=1: all 64 squares orange, every filter sees all 8 channels. Right, groups=4: four orange blocks of 2 by 2 on the diagonal, 16 squares: each filter sees only the 2 channels of its own group."}),
        mo.md(r"""
        **The ResNeXt block.** ResNeXt's paper starts from 32 paths of the same shape side by side in a residual block: each path is a 1×1 convolution from 256 channels to 4, a 3×3 convolution from 4 to 4, and a 1×1 convolution from 4 back to 256. The block adds up what the 32 paths give, and adds its input $x$ to that, as in [ResNet](/course/open/cnn-resnet?lang=en). Because the paths are alike, they fit in three convolutions. The 32 first ones together are one 1×1 convolution from 256 channels to 128. The 32 middle ones, each on its own 4 channels, are one 3×3 convolution from 128 to 128 in 32 groups. And adding up the 32 last ones is one 1×1 convolution from 128 back to 256. The paper calls the number of paths the **cardinality**, and writes a block as C×d, for C paths of d channels: ResNeXt-50 has 32×4d.

        **The same cost, twice as wide.** This block, with 256 channels in and out as in ResNeXt-50, has 70,656 parameters, counted with a bias for each filter, as `nn.Conv2d` makes them. ResNet-50's bottleneck of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), from 256 channels to 64, 64 to 64, and back to 256, has 70,016. In the middle, ResNeXt's block is twice as wide for that cost, 128 channels against 64; without groups, it would have 213,504 parameters. The paper's table 3 compares blocks of about the same cost on ImageNet, from 1 path of 64 channels, ResNet-50 itself, to 32 paths of 4. Its numbers are how many of ImageNet's validation photos the network's first guess got wrong:

        | Block | Paths of channels | Wrong on ImageNet |
        |---|---|---|
        | 1×64d, ResNet-50 | 1 of 64 | 23.9% |
        | 2×40d | 2 of 40 | 23.0% |
        | 4×24d | 4 of 24 | 22.6% |
        | 8×14d | 8 of 14 | 22.3% |
        | 32×4d, ResNeXt-50 | 32 of 4 | 22.2% |
        """),
        L.watch_out("The gain of 1.7 points was measured on ImageNet, 1.28 million photos of 1,000 kinds, with batch normalization after every convolution. A finding holds where it was measured: on a small network and a few thousand photos, the difference can be smaller than what the seed changes. That is what the demo tests. And like the first ResNet of 2015, ResNeXt's paper puts a ReLU after the addition; the course leaves it out, as in [ResNet](/course/open/cnn-resnet?lang=en), so that the input gets through unchanged."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **grouped convolution** | a convolution whose channels and filters are split into groups of the same size: each filter sees only the channels of its own group, so the convolution has `groups` times fewer weights | `nn.Conv2d(128, 128, 3, groups=32)` |
        | **ResNeXt** | a ResNet whose blocks are many thin paths of the same shape, written as a 1×1 convolution, a grouped 3×3 convolution and a 1×1 convolution, with the input added; from 2016 | `ResNeXtBlock(256, 128, 32)` |
        | **cardinality** | the number of paths in a ResNeXt block: the `groups` of its 3×3 convolution. 32×4d is 32 paths of 4 channels | `groups=32` |

        **The demo's network.** The demo trains the network of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), with a ResNeXt block in its middle, on the 64 maps of 8 by 8. It has four blocks to choose from: the paper's 1×64d, 2×40d, 4×24d and 32×4d with all their channels divided by 4, so 1×16d, 2×10d, 4×6d and 32×1d. The paper's 8×14d would give 8 groups of 3.5 channels, so it is left out. Each block has about 4,400 weights, 4,352, 4,360, 4,368 or 4,384, and the whole network 34,090 to 34,154 parameters. 1×16d has exactly the three convolutions of that lesson's bottleneck with 16 in the middle; it adds the input at the end, where the bottleneck has a ReLU. In 32×1d, each 3×3 filter sees a single channel.

        The picture: how many of the first 1,000 test photos of CIFAR-10 the network reads right after 20 epochs on the first 6,000 training photos, a dot for each of 5 seeds, with the course's code. The two rows at the top are from that lesson, without the input added: one 3×3 convolution, 532 to 572, and the bottleneck with 16 in the middle, 521 to 550. The four ResNeXt blocks: 562 to 598, 565 to 580, 572 to 599 and 574 to 584. Exercise 3 asks what the rows say.
        """),
        mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                "plain": "one 3×3 convolution, without x added", "b16": "bottleneck, 16 in the middle, without x added",
                                "1x16": "ResNeXt block 1×16d: 1 group of 16 channels", "2x10": "2×10d: 2 groups of 10",
                                "4x6": "4×6d: 4 groups of 6", "32x1": "32×1d: 32 groups of 1",
                                "mine": "your run", "course": "the course's runs",
                                "alt": "Six rows of open dots, test photos read right of 1,000, one dot for each seed and a short line at their mean. One 3×3 convolution, without x added: 532 to 572. Bottleneck with 16 in the middle, without x added: 521 to 550. ResNeXt block 1×16d: 562 to 598. 2×10d: 565 to 580. 4×6d: 572 to 599. 32×1d: 574 to 584."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a ResNeXt block for `channels` channels, `width` in the middle, in `groups` groups:

    1. A 1×1 convolution from `channels` to `width`, with a ReLU.
    2. A 3×3 convolution from `width` to `width` in `groups` groups, with `padding=1` and a ReLU: each group has `width // groups` channels.
    3. A 1×1 convolution from `width` back to `channels`, without a ReLU.
    4. Give back the input added: `x + y`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes 32 small convolutions, each from 4 channels to 4, and one convolution from 128 channels to 128 in 32 groups, and gives that one the weights and biases of the 32 small ones, one after the other. Then it sends made-up maps of one photo, 128 channels of 14 by 14, both ways: through the small convolutions, each on its own 4 channels, side by side, and through the grouped one at once. It prints the shape and the number of the grouped convolution's weights, and the first 4 channels at one pixel, from both ways. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_groups():
    import torch
    from torch import nn

    torch.manual_seed(0)
    x = torch.rand(1, 128, 14, 14)
    # 32 small convolutions, each from 4 channels to 4
    small = [nn.Conv2d(4, 4, 3, padding=1) for g in range(32)]
    # one convolution from 128 channels to 128, in 32 groups
    conv = nn.Conv2d(128, 128, 3, padding=1, groups=32)  # change: groups=1
    print(conv.weight.shape, conv.weight.numel())
    with torch.no_grad():
        # the weights and biases of the 32 small ones, one after the other
        conv.weight[:] = torch.cat([s.weight for s in small])
        conv.bias[:] = torch.cat([s.bias for s in small])
        # each small convolution on its own 4 channels, side by side
        parts = [small[g](x[:, 4 * g:4 * g + 4]) for g in range(32)]
        print(torch.cat(parts, dim=1)[0, :4, 0, 0])
        print(conv(x)[0, :4, 0, 0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `torch.Size([128, 4, 3, 3]) 4608`: 128 filters, each 4 channels deep. `torch.cat` puts the weights of the 32 small convolutions, each of the shape (4, 4, 3, 3), one after the other along the filters, which gives exactly that shape. Then the same 4 numbers twice, `tensor([ 0.4007, -0.6840, -0.4022,  0.1121])`: the grouped convolution is the 32 small ones side by side. With `groups=1`, the convolution has `torch.Size([128, 128, 3, 3]) 147456` weights, 32 times as many: every filter sees all 128 channels. Then the weights of the small ones do not fit, and PyTorch stops with `The expanded size of the tensor (128) must match the existing size (4) at non-singleton dimension 1`: weights 4 channels deep, where each filter now needs 128.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide the number of groups of the 3×3 convolution in this block of ResNeXt-50, with 256 channels, from 1 to 128. The three bars are the block's three convolutions; the axis stays the same, in thousands of parameters.
    """)
    return


@app.cell(hide_code=True)
def play_groups():
    _labels = {"title": "A block of ResNeXt-50, 256 channels", "x": "parameters (thousands)",
               "reduce": "1×1 convolution, 256 to 128", "conv": "3×3 convolution, 128 to 128, in groups",
               "expand": "1×1 convolution, 128 back to 256",
               "alt": "Three bars of parameters, in thousands, for the chosen number of groups, in orange: the 1×1 convolution from 256 channels to 128, the 3×3 convolution from 128 to 128 in groups, and the 1×1 convolution from 128 back to 256."}
    explore_groups = L.explore("groups-slider", "groups", lambda _v: (C.groups_bars(int(_v), _labels, L.lang), C.groups_sentence(int(_v), L.lang)),
                               start=32, values=C.GROUPS, role="hyper",
                               task="From how many groups on does the 3×3 convolution have fewer parameters than the 1×1 convolution in front of it?")
    explore_groups
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    From 8 groups: then the 3×3 convolution has 18,560 parameters, against 32,896 for the 1×1 convolution in front of it; with 4 groups, it still has 36,992. With 32 groups, as in ResNeXt-50, it has 4,736, and nearly all of the block's 70,656 parameters are in its two 1×1 convolutions: 65,920. That is why ResNeXt can make the middle twice as wide for the same cost. With 128 groups, each filter sees one channel: that is where *Depthwise separable convolutions: MobileNet* starts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_groups():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which groups work

    Six convolutions with `groups`. Which can PyTorch make, and for which does it stop? Sort each one: it works, or PyTorch stops.
    """)
    return


@app.cell(hide_code=True)
def exercise_groups():
    sort_groups = L.sort("groups")
    sort_groups
    return (sort_groups,)


@app.cell(hide_code=True)
def check_groups(sort_groups):
    result_groups = L.check("groups", sort_groups)
    result_groups
    return (result_groups,)


@app.cell(hide_code=True)
def hints_groups():
    L.hints(
        "groups",
        [
            "In `nn.Conv2d(c_in, c_out, k, groups=G)`, both the channels that come in and the filters are split into $G$ groups of the same size. Work out $c_{in} / G$ and $c_{out} / G$ for each.",
            "Three work and three stop. A group of a single channel is fine; half a channel or half a filter is not.",
        ],
        why="**It works:** `nn.Conv2d(128, 128, 3, groups=32)`, 128 / 32 = 4 channels and 4 filters per group; `nn.Conv2d(64, 64, 3, groups=64)`, 1 and 1; `nn.Conv2d(96, 64, 1, groups=8)`, 96 / 8 = 12 channels and 64 / 8 = 8 filters. **PyTorch stops:** `nn.Conv2d(128, 100, 3, groups=32)`, 100 / 32 = 3.125 filters; `nn.Conv2d(128, 128, 3, groups=3)`, 128 / 3 is not a whole number; `nn.Conv2d(64, 32, 3, groups=64)`, half a filter per group.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_resnext():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A ResNeXt block

        Complete `ResNeXtBlock`. Its arguments are the channels that come in and go out, `channels`; the channels in the middle, `width`; and the number of groups, `groups`. `reduce` and `expand`, and the first lines of `forward`, are in the box. Make `conv`, a 3×3 convolution from `width` to `width` in `groups` groups that keeps the height and width, and let `forward` give back the input added. The check builds your block, with the numbers of a block of ResNeXt-50 with 256 channels and with made-up ones, and compares what comes out, and how many parameters it has, with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/resnext.py`.
        """),
        L.task("Make `conv`, and let `forward` give back the input added."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resnext():
    box_resnext = L.code_box("resnext", lines=19)
    box_resnext
    return (box_resnext,)


@app.cell(hide_code=True)
def check_resnext(box_resnext):
    result_resnext = L.check("resnext", box_resnext.value)
    result_resnext
    return (result_resnext,)


@app.cell(hide_code=True)
def hints_resnext():
    L.hints(
        "resnext",
        [
            "`nn.Conv2d` takes the number of groups as the argument `groups`. Which padding keeps the height and width for 3 by 3?",
            "`self.conv = nn.Conv2d(width, width, 3, padding=1, groups=groups)`. The last line is step 4 of the recipe.",
        ],
        why="`conv` is `nn.Conv2d(width, width, 3, padding=1, groups=groups)`, and `forward` ends with `return x + y`. ResNeXt's own blocks have batch normalization after each convolution, and a ReLU after the addition, as the first ResNet had.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains the demo's network, with your `ResNeXtBlock` in its middle, on the first 6,000 training photos of CIFAR-10, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Choose how its block splits the middle into groups: 1 group of 16 channels, 2 of 10, 4 of 6 or 32 of 1. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10 and exercise 2, and takes a few minutes. While it trains, a chart shows the validation loss after every epoch. In the picture after, the gray line is the validation loss of the course's run with one 3×3 convolution in the middle and seed 0. Then you can rewind through your run, and see your run next to the course's.
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
    variant_choice = mo.ui.radio(options={"1 group of 16 channels, 1×16d": "1x16", "2 groups of 10, 2×10d": "2x10",
                                          "4 groups of 6, 4×6d": "4x6", "32 groups of 1, 32×1d": "32x1"},
                                 value="32 groups of 1, 32×1d", label="The 3×3 convolution in the middle")
    train_button = mo.ui.run_button(label="Train the network")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_resnext(cifar, train_button, variant_choice, result_resnext):
    cifar, result_resnext  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the network.*")
    else:
        _variant = variant_choice.value
        _names = {"1x16": "1×16d", "2x10": "2×10d", "4x6": "4×6d", "32x1": "32×1d"}
        _with = {"1x16": "in 1 group of 16 channels", "2x10": "in 2 groups of 10 channels", "4x6": "in 4 groups of 6 channels", "32x1": "in 32 groups of 1 channel"}
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your network on CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "The validation loss of the network on the 1,000 test photos after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**ResNeXt block {_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**ResNeXt block {_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `ResNeXtBlock` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `ResNeXtBlock` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "resnext": "*Your `ResNeXtBlock` in `work/projects/cnn/resnext.py` stops with an error, or does not give back 64 maps of 8 by 8. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "cnn": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give what it should. Did you change its file? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "one 3×3 convolution",
                       "alt": "The loss of the network with the ResNeXt block after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's run with one 3×3 convolution in the middle and seed 0, which falls to epoch 6 and then rises."}
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
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""With your ResNeXt block {_with[_variant]}, the network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters, {C.grouped(_result['block_params'], L.lang)} of them in its block.{_turn} Over 5 seeds, with the course's code, this block: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos, with the lowest validation loss after epoch {_be[0]} to {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                        "plain": "one 3×3 convolution, without x added", "b16": "bottleneck, 16 in the middle, without x added",
                                        "1x16": "ResNeXt block 1×16d: 1 group of 16 channels", "2x10": "2×10d: 2 groups of 10",
                                        "4x6": "4×6d: 4 groups of 6", "32x1": "32×1d: 32 groups of 1",
                                        "mine": "your run", "course": "the course's runs",
                                        "alt": "Six rows of test photos read right, of 1,000: open dots for the course's runs over 5 seeds, with one 3×3 convolution and with the bottleneck with 16 in the middle, both without x added, and with the four ResNeXt blocks, with a short line at the mean of each row; and in the row of your block a filled dot for your run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Exercise 3. What the groups do here
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "groups-runs",
        question="Over 5 seeds, after 20 epochs, each network with about 34,100 parameters. With the ResNeXt block in 1 group of 16 channels, it reads 562 to 598 test photos right; in 2 groups of 10, 565 to 580; in 4 groups of 6, 572 to 599; in 32 groups of 1, 574 to 584. The bottleneck with 16 in the middle of [1×1 convolutions](/course/open/cv-1x1-convolution?lang=en), without the input added, reads 521 to 550. What do the numbers say?",
        options={
            "more": "More groups read more photos right: 32 groups of 1 is best",
            "one": "One group is best: there, every filter sees all 16 channels",
            "right": "The four blocks overlap, so here more groups give no clear gain; the residual block does: each of the four reads more right than the bottleneck without the input added",
        },
        correct="right",
        why={
            "more": "The block with 32 groups of 1 reads 574 to 584 right, 580 on average; with 1 group of 16, it reads 585 on average, and with 4 groups of 6, 584. The highest run of all, 599, is with 4 groups. With 5 seeds each, the four ranges overlap: from these runs, you cannot tell the blocks apart.",
            "one": "The block with 1 group of 16 has the highest average, 585, but the one with 4 groups of 6 has 584, and the five runs with 1 group of 16 go from 562 to 598: the seed alone moves a block's count by 10 to 36. The ranges overlap: from these runs, you cannot tell the blocks apart.",
            "right": "With the input added, the four blocks read 562 to 599 test photos right; the bottleneck without it, 521 to 550: every run of the four beats every run of the bottleneck. For 1×16d these are the same three convolutions, with the input added at the end and without the ReLU after `expand`. With the input added, the network also learns its training photos, 5,938 to 5,999 of 6,000, against 5,077 to 5,510. Between the four blocks, the averages are 585, 570, 584 and 580: they do not rise or fall with the number of groups. The paper's gain, 1.7 points on ImageNet, would be 17 of these 1,000 photos; here 32 groups read 5 fewer than one group on average, and the seed moves the runs of one block by 10 to 36: these runs show no such gain, in a network without batch normalization on 6,000 small photos. What the groups do give here: a wider middle for the same weights, 32 channels in 32 groups against 16 in one.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A ResNeXt block is the residual block of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en) with the bottleneck of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), made twice as wide in the middle by a grouped convolution; its 32 paths side by side are the branches of [Inception: filters of several sizes side by side](/course/open/cv-inception?lang=en), all of one shape. You counted its weights with the shapes of [Convolution in detail](/course/open/cnn-conv-formal?lang=en), put small convolutions side by side with `torch.cat` of [PyTorch tensors](/course/open/torch-tensors?lang=en), and trained the block with the Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) on the photos of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en). Your project `cnn` now has `resnext.py`. In the next lesson, *DenseNet: every layer sees all the layers before it*, a block does not add its input but keeps it: every layer gets the maps of all the layers before it, stacked along the channels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_groups, result_resnext, choice_runs):
    L.wrap_up(result_groups, result_resnext, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
