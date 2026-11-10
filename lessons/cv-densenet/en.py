# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-densenet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-densenet", "en")
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
        question="`x` is 64 channels of 8 by 8 and `y` is 12 channels of 8 by 8, for a batch of 1 photo. What shape does `torch.cat([x, y], dim=1)` give?",
        options={
            "right": "(1, 76, 8, 8)",
            "plus": "(1, 64, 8, 8): the 12 channels of `y` are added to the first 12 of `x`",
            "stops": "None: PyTorch stops, because `x` and `y` have different numbers of channels",
        },
        correct="right",
        why={
            "right": "Along `dim=1`, the channels, the 12 maps of `y` come after the 64 of `x`: 64 + 12 = 76. Today every layer of a block does this with the maps it makes.",
            "plus": "Adding number by number is `+`, and it would stop here: 12 channels do not fit on 64. `torch.cat` puts the maps after each other. See [Inception](/course/open/cv-inception?lang=en).",
            "stops": "Along `dim=1`, `torch.cat` needs the same size in the other dimensions only: the photos, the height and the width. The channels may differ, as with the branches of Inception, which gave 16, 32, 8 and 8 maps. See [Inception](/course/open/cv-inception?lang=en).",
        },
    )
    quick_cat
    return (quick_cat,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="A residual block gets $x$, 64 channels of 8 by 8, and gives back $x + f(x)$. What does the layer after the block get?",
        options={
            "right": "64 channels of 8 by 8, in which the numbers of $x$ and $f(x)$ are added together",
            "side": "128 channels: the 64 of $x$, and the 64 of $f(x)$ after them",
            "only": "Only $f(x)$: the block uses $x$ inside, and then lets it go",
        },
        correct="right",
        why={
            "right": "`+` adds number by number: what comes out has the shape of $x$, and $x$ and $f(x)$ are mixed in it. The layer after the block cannot see $x$ on its own any more. Today's block keeps them apart.",
            "side": "Putting maps after each other along the channels is `torch.cat`. `+` adds them number by number, so the sum has the shape of $x$. See [ResNet](/course/open/cnn-resnet?lang=en).",
            "only": "The block keeps $x$: it adds it to $f(x)$, so that the gradient has a short way back. But it gives the sum, not $x$ on its own. See [ResNet](/course/open/cnn-resnet?lang=en).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en), a block adds its input to what its layers make: $x + f(x)$. That gives the gradient a short way back, but the layer after the block gets only the sum, in which the input and the new maps are mixed. In 2016, Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Weinberger kept them apart instead. In their DenseNet, every layer of a block gets the maps of the block's input and of all the layers before it, put after each other along the channels with `torch.cat`, as the branches of [Inception: filters of several sizes side by side](/course/open/cv-inception?lang=en) were. Because every layer sees everything before it, each layer can be thin: in the paper's small networks for CIFAR-10, it adds only 12 maps of its own. A DenseNet of theirs with 100 layers and 0.8 million parameters got 4.51% of CIFAR-10's test photos wrong, about as few as a ResNet of 1,001 layers with 10.2 million, 4.62%. On ImageNet, their DenseNet-201 with 20 million parameters made about as many mistakes as a ResNet of 101 layers with more than 40 million. The paper won the best paper award at the CVPR conference in 2017. Today you count what the layers of a dense block get and give, build one, and test in a CNN on CIFAR-10 what its connections do.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_channels():
    guess_channels = L.guess(
        "dense-channels",
        question="A dense block gets 64 channels of 8 by 8 and has 4 layers. Each layer gets the maps of the block's input and of every layer before it, and makes 12 maps of its own, which go after them. How many channels does the block give back?",
        options={
            "same": "64: as many as it gets, like a residual block",
            "last": "12: the maps of its last layer",
            "right": "112",
        },
        reveal="**112.** The block keeps everything: the 64 channels of its input, and the 12 maps of each of its 4 layers, 64 + 4 × 12 = 112. Its first layer gets 64 channels, the second 76, the third 88 and the fourth 100: each layer gets 12 more than the one before. A residual block would give back 64.",
    )
    guess_channels
    return (guess_channels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Dense connections.** A dense block is a row of layers in which every layer gets the maps of the block's input and of all the layers before it, put after each other along the channels with `torch.cat`, and makes $k$ new maps of its own. The paper calls $k$ the **growth rate**: with every layer, the block's maps grow by $k$ channels. If the input has $k_0$ channels, layer $l$ gets $k_0 + k(l - 1)$ of them, and after $L$ layers the block gives back $k_0 + kL$. Nothing is added up or thrown away: the input stays at the front, unchanged, and the maps of every layer stay where they were put. So every layer has a connection from the input and from each layer before it: in a block of $L$ layers, $L(L + 1) / 2$ connections, where a plain row of layers has $L$. That is where the name comes from.

        The picture: a dense block with growth 12 on 64 channels, with 4 layers and so 10 connections; the arrows into the last layer are darker. Under each layer, how many channels it gets and, in magenta, the 12 maps it makes.
        """),
        C.connections_picture({"title": "A dense block: 4 layers, growth 12", "input": "input x", "layer": "layer {l}",
                               "gets": "gets {n}", "gives": "gives {n}",
                               "alt": "Five boxes in a row: input x, which gives 64 channels, and layers 1 to 4. Every box has an arrow to every later layer: 10 arrows. Layer 1 gets 64 channels, layer 2 76, layer 3 88 and layer 4 100; each layer gives 12 maps. The four arrows into layer 4 are darker."}),
        mo.md(r"""
        **The transition layer.** The channels grow with every layer, and `torch.cat` needs maps of the same height and width, so a block cannot make its maps smaller. A DenseNet is therefore a few dense blocks in a row, with a **transition layer** between each two: a 1×1 convolution, which mixes the channels, as in [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), and average pooling of 2 by 2, which halves the height and width. The paper's DenseNet-BC lets each transition give only half the channels it gets, and puts a 1×1 convolution to $4k$ channels in front of every 3×3 convolution, a bottleneck: then a 3×3 convolution gets $4k$ channels, however many came before. Its layers have batch normalization and a ReLU before every convolution, as the ResNets of 2016 in the table below.

        The paper's table 2 compares networks on CIFAR-10, trained on flipped and shifted photos. Its numbers are the share of the 10,000 test photos the network got wrong:

        | Network | Parameters | Wrong on CIFAR-10 |
        |---|---|---|
        | ResNet, 164 layers | 1.7 million | 5.46% |
        | ResNet, 1,001 layers | 10.2 million | 4.62% |
        | DenseNet, $k = 12$, 40 layers | 1.0 million | 5.24% |
        | DenseNet-BC, $k = 12$, 100 layers | 0.8 million | 4.51% |
        | DenseNet-BC, $k = 40$, 190 layers | 25.6 million | 3.46% |
        """),
        L.watch_out("A dense block has nothing to do with what some libraries, such as Keras, call a dense layer: that is a linear layer, `nn.Linear`. Dense here means that every layer of the block is connected to every later one. And the paper's numbers come from networks with batch normalization, flipped and shifted training photos and 300 epochs. A finding holds where it was measured: what dense connections do in a small network without those, the demo has to show."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **dense block** | a row of layers in which every layer gets the maps of the block's input and of all the layers before it, put after each other along the channels, and adds maps of its own | `DenseBlock(64, 12, 4)` |
        | **growth rate** | how many maps each layer of a dense block adds, $k$: layer $l$ gets $k_0 + k(l - 1)$ channels | `growth=12` |
        | **transition layer** | the layer between two dense blocks: a 1×1 convolution that mixes the channels, in DenseNet-BC into half as many, and pooling that makes the maps smaller | `nn.Conv2d(112, 64, 1)` |
        | **DenseNet** | a CNN of dense blocks with transition layers between them; from 2016 | — |

        **The demo's network.** The demo trains the network of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en), with a dense block in its middle, on the 64 maps of 8 by 8, and after it a transition: a 1×1 convolution from all the block's channels back to 64, with a ReLU. The pooling after it is the network's own. It has three dense blocks to choose from, all giving back 112 channels: growth 24 with 2 layers, growth 12 with 4, and growth 6 with 8. With its transition, the middle has 40,112, 42,704 or 44,000 parameters. To see what the connections themselves do, the fourth choice is the course's chain: 4 layers of 3×3 convolutions with 30 channels, each getting only the maps of the layer before, then a transition from 30 channels back to 64. With 43,684 parameters, it has about as many as growth 12 with 4 layers, and it is as deep. One 3×3 convolution from 64 channels to 64, as in that lesson, has 36,928.

        The picture: how many of the first 1,000 test photos of CIFAR-10 the network reads right after 20 epochs on the first 6,000 training photos, a dot for each of 5 seeds, with the course's code. The row at the top is from that lesson: one 3×3 convolution, 532 to 572. The chain: 452 to 515. The dense blocks with growth 24, 12 and 6: 553 to 584, 546 to 573 and 557 to 600. Exercise 3 asks what the rows say.
        """),
        mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                "plain": "one 3×3 convolution", "chain": "chain: 4 layers of 30, each sees the one before",
                                "k24": "dense block, growth 24, 2 layers", "k12": "growth 12, 4 layers", "k6": "growth 6, 8 layers",
                                "mine": "your run", "course": "the course's runs",
                                "alt": "Five rows of open dots, test photos read right of 1,000, one dot for each seed and a short line at their mean. One 3×3 convolution: 532 to 572. Chain of 4 layers of 30: 452 to 515. Dense block with growth 24 and 2 layers: 553 to 584. Growth 12 and 4 layers: 546 to 573. Growth 6 and 8 layers: 557 to 600."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a dense block for `channels` channels, with growth `growth` and `layers` layers:

    1. In `__init__`, make layer `i` (0, 1, 2, ...) a 3×3 convolution from all the maps it gets, the block's input and the `growth` maps of each layer before it, to `growth` maps, with `padding=1`, and keep the layers in an `nn.ModuleList`.
    2. In `forward`, send `x` through each layer and a ReLU: that gives `y`, `growth` new maps.
    3. Put `y` after `x` along the channels: `x = torch.cat([x, y], dim=1)`.
    4. After the last layer, give back `x`: `channels + layers * growth` channels.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes the four layers of a dense block with growth 12 on 64 channels, and sends made-up maps of one photo, 64 channels of 8 by 8, through them. After each layer it prints the shape of the layer's weights and the shape of `x`. Then it prints the first 3 numbers of the input at the top-left pixel, before the block and after it, and how many weights the four layers have together. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_dense():
    import torch
    from torch import nn

    torch.manual_seed(0)
    x = torch.rand(1, 64, 8, 8)
    x0 = x
    growth = 12  # change: growth = 24
    # layer i gets the 64 channels of the input and the maps of the i layers before it
    layers = []
    for i in range(4):
        layers.append(nn.Conv2d(64 + i * growth, growth, 3, padding=1))
    with torch.no_grad():
        for layer in layers:
            y = torch.relu(layer(x))
            # put the new maps after all the maps so far, along the channels
            x = torch.cat([x, y], dim=1)
            print(list(layer.weight.shape), list(x.shape))
    print(x0[0, :3, 0, 0])
    print(x[0, :3, 0, 0])
    print(sum(layer.weight.numel() for layer in layers))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `[12, 64, 3, 3] [1, 76, 8, 8]`, then `[12, 76, 3, 3] [1, 88, 8, 8]`, `[12, 88, 3, 3] [1, 100, 8, 8]` and `[12, 100, 3, 3] [1, 112, 8, 8]`: every layer has 12 filters, each as deep as the channels it gets, and after it `x` has 12 channels more. Then twice `tensor([0.4963, 0.7745, 0.3251])`: the input is still at the front of `x`, unchanged. And `35424`, the weights of the four layers, 12 × 9 × (64 + 76 + 88 + 100). With `growth = 24`, the layers get 64, 88, 112 and 136 channels, and the block gives back `[1, 160, 8, 8]`: 64 + 4 × 24. Its layers then have `86400` weights, more than twice as many: every layer has twice as many filters, and gets more channels too.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide the number of layers of a dense block with growth 12 on 64 channels, from 1 to 8. Each bar is what a layer gets: the block's input in gray, and in magenta the 12 maps of each layer before it. In orange, the layer's parameters. The last bar is what the block gives back; the axis stays the same.
    """)
    return


@app.cell(hide_code=True)
def play_layers():
    _labels = {"title": "A dense block with growth 12 on 64 channels", "x": "channels",
               "layer": "layer {l} gets {n}", "out": "the block gives back {n}",
               "alt": "A bar for each layer of the dense block, the channels it gets: the 64 of the block's input in gray, and 12 in magenta for each layer before it, with the layer's parameters in orange; and a last bar for the channels the block gives back."}
    explore_layers = L.explore("layers-slider", "layers", lambda _v: (C.stack_picture(int(_v), _labels, L.lang), C.stack_sentence(int(_v), L.lang)),
                               start=4, values=C.LAYERS, role="hyper",
                               task="From how many layers on do the block's layers have more parameters together than one 3×3 convolution from 64 channels to 64?")
    explore_layers
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    From 5 layers: then they have 47,580 parameters together, against 36,928; with 4 layers, 35,472. Each layer gets 12 channels more than the one before, so each costs more than the one before: the first 6,924 parameters, the eighth 15,996. With 8 layers, the block's layers have 91,680 parameters, and the block has 36 connections. The cost grows faster than the number of layers: that is why DenseNet-BC puts a 1×1 convolution to $4k$ channels in front of every 3×3 convolution, and why its transitions make the channels fewer between the blocks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_blocks():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Adding or keeping

    Six sentences about a block. Which describe a residual block of ResNet that gives back $x + f(x)$, and which a dense block of DenseNet? Sort each one.
    """)
    return


@app.cell(hide_code=True)
def exercise_blocks():
    sort_blocks = L.sort("blocks")
    sort_blocks
    return (sort_blocks,)


@app.cell(hide_code=True)
def check_blocks(sort_blocks):
    result_blocks = L.check("blocks", sort_blocks)
    result_blocks
    return (result_blocks,)


@app.cell(hide_code=True)
def hints_blocks():
    L.hints(
        "blocks",
        [
            "A residual block gives back $x + f(x)$; a dense block sets `x = torch.cat([x, y], dim=1)` after every layer. What does each do to the number of channels?",
            "Three sentences belong to each block. Ask for each sentence: does the block add up, or put after?",
        ],
        why="**A residual block:** it gives back as many channels as it gets, it adds what its layers make to its input, number by number, and what its layers make must have the shape of its input. **A dense block:** it gives back more channels than it gets, it puts the maps of each layer after the maps it got, and its last layer gets the maps of every layer before it, unchanged.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dense():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A dense block

        Complete `DenseBlock`. Its arguments are the channels that come in, `channels`; how many maps each layer adds, `growth`; and the number of layers, `layers`. The loop in `__init__` and the first lines of `forward` are in the box. Make layer `i`, a 3×3 convolution from all the maps it gets to `growth` maps, that keeps the height and width, and let `forward` put `y` after `x`. The check builds your block, with growth 12 and 4 layers on 64 channels and with made-up numbers, and compares what comes out, and how many parameters it has, with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/dense.py`.
        """),
        L.task("Make the convolution of layer `i`, and let `forward` put `y` after `x`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dense():
    box_dense = L.code_box("dense", lines=18)
    box_dense
    return (box_dense,)


@app.cell(hide_code=True)
def check_dense(box_dense):
    result_dense = L.check("dense", box_dense.value)
    result_dense
    return (result_dense,)


@app.cell(hide_code=True)
def hints_dense():
    L.hints(
        "dense",
        [
            "Layer `i` gets the block's input, `channels`, and the `growth` maps of each of the `i` layers before it. Which padding keeps the height and width for 3 by 3?",
            "`self.layers.append(nn.Conv2d(channels + i * growth, growth, 3, padding=1))`. The new `x` is step 3 of the recipe.",
        ],
        why="Layer `i` is `nn.Conv2d(channels + i * growth, growth, 3, padding=1)`, and `forward` sets `x = torch.cat([x, y], dim=1)`. DenseNet's own layers have batch normalization and a ReLU before their convolution, and in DenseNet-BC a 1×1 convolution to 4 × `growth` channels in front of it.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains the demo's network, with your `DenseBlock` and its transition in its middle, on the first 6,000 training photos of CIFAR-10, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Choose the middle: a dense block with growth 24 and 2 layers, 12 and 4, or 6 and 8, or the course's chain of 4 layers of 30 channels without dense connections, which needs no block of yours. Where a module of yours from an earlier lesson is missing or broken, the course's stands in. It needs CIFAR-10, and for a dense block exercise 2, and takes a few minutes. While it trains, a chart shows the validation loss after every epoch. In the picture after, the gray line is the validation loss of the course's run with one 3×3 convolution in the middle and seed 0. Then you can rewind through your run, and see your run next to the course's.
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
    variant_choice = mo.ui.radio(options={"dense block, growth 24, 2 layers": "k24", "dense block, growth 12, 4 layers": "k12",
                                          "dense block, growth 6, 8 layers": "k6", "chain of 4 layers of 30, without dense connections": "chain"},
                                 value="dense block, growth 12, 4 layers", label="The middle of the network")
    train_button = mo.ui.run_button(label="Train the network")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_dense(cifar, train_button, variant_choice, result_dense):
    cifar, result_dense  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the network.*")
    else:
        _variant = variant_choice.value
        _names = {"k24": "Dense block, growth 24", "k12": "Dense block, growth 12", "k6": "Dense block, growth 6", "chain": "Chain of 4 layers"}
        _with = {"k24": "With your dense block with growth 24 and 2 layers", "k12": "With your dense block with growth 12 and 4 layers",
                 "k6": "With your dense block with growth 6 and 8 layers", "chain": "With the course's chain of 4 layers of 30 channels"}
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your network on CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "The validation loss of the network on the 1,000 test photos after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**{_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `DenseBlock` is in your project, and this cell uses it. The chain needs no block of yours.*")
        elif isinstance(_result, str):
            _growth, _layers = C.VARIANTS.get(_variant, (0, 0))
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `DenseBlock` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "dense": f"*Your `DenseBlock` in `work/projects/cnn/dense.py` stops with an error, or does not give back the channels it should: with 64 channels in, growth {_growth} and {_layers} layers, {C.channels_out(64, _growth, _layers)} maps of 8 by 8. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
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
                mo.md(f"""{_with[_variant]}, the network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters, {C.grouped(_result['middle_params'], L.lang)} of them in its middle.{_turn} Over 5 seeds, with the course's code, this middle: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos, with the lowest validation loss after epoch {_be[0]} to {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Test photos read right after 20 epochs", "x": "read right, of 1,000",
                                        "plain": "one 3×3 convolution", "chain": "chain: 4 layers of 30, each sees the one before",
                                        "k24": "dense block, growth 24, 2 layers", "k12": "growth 12, 4 layers", "k6": "growth 6, 8 layers",
                                        "mine": "your run", "course": "the course's runs",
                                        "alt": "Five rows of test photos read right, of 1,000: open dots for the course's runs over 5 seeds, with one 3×3 convolution, with the chain of 4 layers of 30, and with the three dense blocks, with a short line at the mean of each row; and in the row of your middle a filled dot for your run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Exercise 3. What the connections do here
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "dense-runs",
        question="Over 5 seeds, after 20 epochs. With a dense block in the middle, with growth 24 and 2 layers, 12 and 4, or 6 and 8, the network reads 553 to 584, 546 to 573 and 557 to 600 test photos right. With the chain of 4 layers of 30 channels, each getting only the maps of the layer before, as deep as growth 12 with 4 layers and with about as many parameters, it reads 452 to 515. What do the numbers say?",
        options={
            "deeper": "More layers with a smaller growth rate read more right: growth 6 with 8 layers is best",
            "weights": "The chain reads fewer photos right because it has fewer parameters",
            "right": "The connections make the difference: every run with a dense block beats every run of the chain, while the three dense blocks overlap",
        },
        correct="right",
        why={
            "deeper": "Growth 6 with 8 layers has the highest average, 572, and the highest run, 600. But growth 24 with 2 layers reads 570 on average, and growth 12 with 4 layers, between the two in depth, 561: the averages do not rise with the layers, and with 5 seeds each the three ranges overlap. From these runs, you cannot tell the three blocks apart.",
            "weights": "The chain's middle has 43,684 parameters, more than the dense block with growth 12 and 4 layers, 42,704, and it is as deep: 4 layers of 3×3. Yet it reads 452 to 515 right, against 546 to 573. Look at what its layers get.",
            "right": "Every run with a dense block, 546 to 600, beats every run of the chain, 452 to 515, including growth 12 with 4 layers, which is as deep and has slightly fewer parameters. The chain also learns its training photos badly: 3,908 to 4,727 of 6,000 after 20 epochs, against 5,918 to 5,990, with its lowest validation loss only after epoch 12 to 15. In a dense block, every layer gets the input unchanged, and the transition gets the input and the maps of every layer; the chain squeezes everything through 30 channels and four layers. The paper's explanation is short ways for the gradient, as in ResNet; these runs show the difference, not its cause. Between the three dense blocks, the averages are 570, 561 and 572, with no order in them. On average they read a few more photos right than one 3×3 convolution, 549, but its runs, 532 to 572, overlap with theirs.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A dense block keeps what the residual block of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en) adds up: it puts the maps of its layers after each other with `torch.cat`, as the branches of [Inception: filters of several sizes side by side](/course/open/cv-inception?lang=en), and its transition layer is a 1×1 convolution of [1×1 convolutions: mixing the channels](/course/open/cv-1x1-convolution?lang=en). You counted the weights of its layers with the shapes of [Convolution in detail](/course/open/cnn-conv-formal?lang=en), kept its layers in an `nn.ModuleList` as in [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en), and trained it with the Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) on the photos of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en). Your project `cnn` now has `dense.py`. In the next lesson, *Depthwise separable convolutions: MobileNet*, a convolution gives every channel a filter of its own, and a 1×1 convolution mixes the channels after it: the work of a 3×3 convolution, with far fewer multiplications.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_blocks, result_dense, choice_runs):
    L.wrap_up(result_blocks, result_dense, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
