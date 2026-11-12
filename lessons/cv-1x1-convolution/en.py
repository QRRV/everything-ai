# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-1x1-convolution. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-1x1-convolution", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="The first layer of your ColorCNN was `nn.Conv2d(3, 8, 3, padding=1)`. How many weights does one of its filters have?",
        options={
            "nine": "9: a piece of 3 by 3",
            "right": "27: a piece of 3 by 3 of each of the 3 colors",
            "all": "216: 8 filters of 27",
        },
        correct="right",
        why={
            "nine": "9 is a piece of 3 by 3 of one channel. A filter of a color photo looks at all 3 channels at once: a piece of 3 by 3 of the red, of the green and of the blue. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "all": "216 = 8 × 27 are the weights of all 8 filters together. One filter has 3 × 3 × 3 = 27, and a bias. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "right": "3 channels × 3 × 3 = 27 weights, and a bias. Today's filters are the smallest there are, 1 by 1: on a color photo, 3 weights, one for each channel.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="A batch of one color photo of 32 by 32, shape (1, 3, 32, 32), goes through `nn.Conv2d(3, 8, 3, padding=1)`. What shape comes out?",
        options={
            "same": "(1, 3, 32, 32)",
            "no_pad": "(1, 8, 30, 30)",
            "right": "(1, 8, 32, 32)",
        },
        correct="right",
        why={
            "same": "A convolution gives one channel for each filter: 8 filters, 8 channels, however many went in. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "no_pad": "That is without padding: a filter of 3 by 3 fits 30 times across 32. `padding=1` adds a border of one pixel, so it fits 32 times. See [Convolution in detail](/course/open/cnn-conv-formal?lang=en).",
            "right": "8 filters give 8 channels, and the padding keeps 32 by 32. A filter of 1 by 1 keeps the size without any padding: it fits on every pixel.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en), the residual connection got a convolution of 1 by 1 where the channels of $x$ and $f(x)$ did not fit. A filter of 1 by 1 sounds useless: it sees one pixel, so it cannot find an edge or a shape. But a pixel deep in a CNN is not one number. It has a number in each channel, 64 or 512 of them, and a filter of 1 by 1 mixes those into new ones. In 2013, Min Lin, Qiang Chen and Shuicheng Yan put such convolutions after the larger ones in their network, *Network In Network*. In 2014, GoogLeNet used them to make the channels fewer before its expensive convolutions of 3 by 3 and 5 by 5, so that it could be deeper and wider for the same work, and won ImageNet's contest. In 2015, ResNet-50 and the deeper ResNets built every block that way: a bottleneck. Today you mix the colors of a photo with a 1×1 convolution, count what a bottleneck saves, build one, and put it in a CNN on CIFAR-10.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_bottleneck():
    guess_bottleneck = L.guess(
        "bottleneck-params",
        question="In ResNet-50, many blocks get and give 256 channels. One convolution of 3 by 3 from 256 channels to 256 would have 590,080 parameters. ResNet-50 uses three convolutions instead: 1 by 1 from 256 channels to 64, 3 by 3 from 64 to 64, and 1 by 1 from 64 back to 256. How many parameters do those three have?",
        options={
            "more": "More: three convolutions instead of one",
            "half": "About half",
            "right": "About an eighth",
        },
        reveal="**About an eighth: 70,016.** The 1×1 convolution from 256 to 64 has 64 × (256 + 1) = 16,448; the 3×3 from 64 to 64, 64 × (64 × 9 + 1) = 36,928; and the 1×1 from 64 back to 256, 256 × (64 + 1) = 16,640. Together 8.4 times fewer than 590,080. For each pixel they do 69,632 multiplications, against 589,824; and about as many as the 73,728 of the block of two 3×3 convolutions of 64 channels in the smaller ResNets, such as ResNet-34: the same work, on maps with 4 times as many channels. Counted with a bias for each filter, as `nn.Conv2d` makes them; ResNet leaves the biases out, because batch normalization comes after each convolution.",
    )
    guess_bottleneck
    return (guess_bottleneck,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One pixel, all its channels.** A 1×1 convolution has filters of 1 by 1. At each pixel, a filter takes the number of that pixel in every channel that comes in, multiplies each by its own weight, adds them up with its bias, and that is the number of its channel at that pixel. With $c_{in}$ channels in and $c_{out}$ filters, that is a linear layer from $c_{in}$ numbers to $c_{out}$, the same one at every pixel. The rows and columns stay as they are; only the number of channels changes.

        The picture: the landscape of [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en), and what two 1×1 convolutions from 3 channels to 1 make of it, drawn with the lowest number black and the highest white. The weights 0.299, 0.587 and 0.114 for red, green and blue give a gray photo: they say how bright each color looks to us, green the most and blue the least, and they are a standard for televisions. The weights 1, 0 and −1 give red minus blue: light where a pixel has more red than blue, as in the yellow sun, and dark where it has more blue, as in the sky.
        """),
        mo.Html(C.mix_picture({"photo": "photo, 3 channels", "gray": "0.299, 0.587, 0.114", "red_blue": "1, 0, −1",
                               "alt": "Three pictures of the same landscape of 24 by 40 pixels: in color; in gray, made with the weights 0.299, 0.587 and 0.114; and red minus blue, made with the weights 1, 0 and −1, with the sun light and the sky dark."})),
        mo.md(r"""
        **Its parameters.** A 1×1 convolution from $c_{in}$ channels to $c_{out}$ has $c_{out} \times (c_{in} + 1)$ parameters: a weight for each channel that comes in, and a bias, for each filter. A 3×3 convolution has $c_{out} \times (9\,c_{in} + 1)$: nine times as many weights. Both grow with $c_{in} \times c_{out}$, so a convolution between many channels is expensive, and a 3×3 one most of all.

        **A bottleneck.** So put the expensive 3×3 convolution in a narrow place. A bottleneck around $c$ channels has three convolutions, each with a ReLU after it: a 1×1 convolution from $c$ channels to $m$, fewer; the 3×3 convolution from $m$ to $m$; and a 1×1 convolution from $m$ back to $c$. What comes out has as many channels as what went in, so it fits where one 3×3 convolution from $c$ to $c$ was.
        """),
        L.watch_out("A 1×1 convolution never looks at a neighbor: what it gives at a pixel comes from that pixel alone. A stack of only 1×1 convolutions finds no edges, no shapes, nothing that spans two pixels. That is why a bottleneck keeps a 3×3 convolution in its middle: the 1×1 convolutions around it only mix channels."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **1×1 convolution** | a convolution with filters of 1 by 1: at each pixel, it mixes the numbers of that pixel's channels into new channels, as a linear layer on every pixel | `nn.Conv2d(c_in, c_out, 1)` |
        | **bottleneck** | a narrow middle: a 1×1 convolution to fewer channels, the expensive convolution on those, and a 1×1 convolution back to as many as before | |

        **The demo's network.** The demo trains a CNN like your ColorCNN, but wider and with one more step: 32 and 64 filters of 3 by 3, each with max pooling of 2 by 2, then a block on the 64 maps of 8 by 8, pooling again, and one linear layer to the 10 logits. The block is one 3×3 convolution from 64 to 64 with a ReLU, 36,928 parameters, or your `Bottleneck` of exercise 2, with 32 or 16 channels in the middle: 13,440 or 4,448 parameters. The whole network then has 66,570, 43,082 or 34,090: with 16 in the middle, about half.

        The picture: how many photos the network reads right after 20 epochs on the first 6,000 training photos of CIFAR-10, of those training photos, filled, and of the first 1,000 test photos, open, a dot for each of 5 seeds, with the course's code. With one 3×3 convolution, it reads 5,999 to 6,000 of its training photos right, and 532 to 572 test photos; its validation loss is lowest after epoch 4 to 6, and then rises again. With 32 in the middle: 5,333 to 5,673 training photos and 527 to 553 test photos. With 16 in the middle: 5,077 to 5,510 training photos and 521 to 550 test photos, with the lowest validation loss after epoch 7 to 12. Exercise 3 asks what the numbers say.
        """),
        mo.Html(C.runs_picture({"title": "Read right after 20 epochs", "x": "read right (%)", "plain": "one 3×3 convolution",
                                "b32": "bottleneck, 32 in the middle", "b16": "bottleneck, 16 in the middle", "mine": "your run",
                                "train": "training", "test": "test",
                                "alt": "Three rows of dots between 40 and 100 percent read right, after 20 epochs. With one 3×3 convolution in the middle: training photos about 100%, test photos about 55%. A bottleneck of 32 in the middle: training about 92%, test about 54%. A bottleneck of 16 in the middle: training about 89%, test about 54%."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To mix the channels of a map with a 1×1 convolution:

    1. Make it with `nn.Conv2d(c_in, c_out, 1)`: a kernel size of 1, no padding needed. The rows and columns stay.
    2. Its `weight` has shape `(c_out, c_in, 1, 1)`: `weight[k, :, 0, 0]` holds the weights of output channel `k`, one for each channel that comes in.
    3. It has $c_{out} \times (c_{in} + 1)$ parameters, against $c_{out} \times (9\,c_{in} + 1)$ for a 3×3 convolution.
    4. For a bottleneck around $c$ channels with $m$ in the middle: a 1×1 convolution from $c$ to $m$, a 3×3 from $m$ to $m$ with padding 1, and a 1×1 from $m$ back to $c$, each with a ReLU after it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads the landscape as a batch of one photo, makes a 1×1 convolution from 3 channels to 1, gives it the weights of a gray value, and prints the shape of its weights, the shapes of the photo and of what comes out, the three colors of a pixel in the sun, row 5 and column 32, and what the convolution makes of that pixel. Then it counts the convolution's parameters. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_mix():
    import matplotlib.pyplot as plt
    import torch
    from torch import nn

    img = plt.imread("lessons/data-images-as-arrays/landscape.png")
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    # 1. a 1×1 convolution from 3 channels to 1
    mix = nn.Conv2d(3, 1, 1)
    # 2. its weights: one for red, one for green and one for blue
    print(mix.weight.shape)
    with torch.no_grad():
        mix.weight[0, :, 0, 0] = torch.tensor([0.299, 0.587, 0.114])  # change: 1.0, 0.0, -1.0
        mix.bias[0] = 0.0
        out = mix(x)
    print(x.shape, out.shape)
    print(x[0, :, 5, 32])
    print(out[0, 0, 5, 32])
    # 3. its parameters: 3 weights and a bias
    print(sum(q.numel() for q in mix.parameters()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `torch.Size([1, 3, 1, 1])` first: one output channel, with a weight for each of the 3 channels that come in, as step 2 of the recipe says. Then `torch.Size([1, 3, 24, 40]) torch.Size([1, 1, 24, 40])`: 3 channels in, 1 out, and the same 24 by 40 pixels. The pixel in the sun is `tensor([0.9804, 0.8431, 0.2353])`: red 0.9804, green 0.8431 and blue 0.2353. The convolution makes `tensor(0.8149)` of it: 0.299 × 0.9804 + 0.587 × 0.8431 + 0.114 × 0.2353 = 0.8149, each channel times its weight, added up. With `1.0, 0.0, -1.0`, it prints `tensor(0.7451)`: red minus blue, high in the yellow sun. Last, `4`: 3 weights and a bias, which the convolution uses at every one of the 24 × 40 = 960 pixels. That is the picture under *The idea*.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide the number of channels in the middle of a bottleneck around 64 channels. The orange bar is its three convolutions, in thousands of parameters, one after the other; the gray bar is one 3×3 convolution from 64 to 64, with a dashed line where it ends.
    """)
    return


@app.cell(hide_code=True)
def play_middle():
    _labels = {"title": "A bottleneck around 64 channels", "x": "parameters (thousands)", "plain": "one 3×3", "bottleneck": "bottleneck",
               "alt": "Two bars of parameters, in thousands: one 3×3 convolution from 64 to 64 channels, 36,928, in gray, with a dashed line where it ends; and in orange the three convolutions of a bottleneck with the chosen number of channels in the middle, one after the other."}
    explore_middle = L.explore("middle", "channels in the middle", lambda _v: (C.bottleneck_picture(int(_v), _labels, L.lang), C.bottleneck_sentence(int(_v), L.lang)),
                               start=16, values=list(range(1, 65)), role="hyper",
                               task="With how many channels in the middle does the bottleneck first have more parameters than the one 3×3 convolution?")
    explore_middle
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 58: with 57 in the middle, it has 36,715 parameters, and with 58, 37,880. The 3×3 convolution in the middle grows with $m \times m$, so the saving grows fast as $m$ gets smaller: with 32 in the middle, 13,440, 36.4% of the 36,928; with 16, 4,448, 12.0%. ResNet-50 keeps a quarter of the channels in the middle: 64 of 256.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one_by_one():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One pixel or more?

    Six things you might want a convolution to do with a photo. Which can a 1×1 convolution do, with the right weights, and which need a larger filter? Put each in its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_one_by_one():
    sort_one_by_one = L.sort("one-by-one")
    sort_one_by_one
    return (sort_one_by_one,)


@app.cell(hide_code=True)
def check_one_by_one(sort_one_by_one):
    result_one_by_one = L.check("one-by-one", sort_one_by_one)
    result_one_by_one
    return (result_one_by_one,)


@app.cell(hide_code=True)
def hints_one_by_one():
    L.hints(
        "one-by-one",
        [
            "For each one, ask: does the new number at a pixel need anything but the old numbers of that same pixel?",
            "It can do three of them, and not the other three. A 1×1 convolution can make any number of new channels from the channels of one pixel, with any weights; it cannot look one pixel further.",
        ],
        why="**A 1×1 convolution can:** a gray photo, with the weights 0.299, 0.587 and 0.114; fewer channels, with 16 filters of 64 weights each; and swapped colors, with a weight of 1 from the old blue to the new red and from the old red to the new blue, 1 from green to green, and 0 everywhere else. **It needs a larger filter:** an edge, a blur and a shift all need the neighbors of a pixel.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bottleneck():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A bottleneck

        Complete `Bottleneck`, a block that fits where a 3×3 convolution from `channels` to `channels` was. Make `reduce`, a 1×1 convolution from `channels` to `middle`, and `expand`, a 1×1 convolution from `middle` back to `channels`; `conv`, the 3×3 convolution between them, is in the box. In `forward`, send `x` through `reduce`, `conv` and `expand`, in that order, each with a ReLU after it, and give back what comes out. The check builds your block, gives it made-up numbers, and compares what comes out, and how many parameters it has, with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/bottleneck.py`.
        """),
        L.task("Make `reduce` and `expand`, and finish `forward`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bottleneck():
    box_bottleneck = L.code_box("bottleneck", lines=18)
    box_bottleneck
    return (box_bottleneck,)


@app.cell(hide_code=True)
def check_bottleneck(box_bottleneck):
    result_bottleneck = L.check("bottleneck", box_bottleneck.value)
    result_bottleneck
    return (result_bottleneck,)


@app.cell(hide_code=True)
def hints_bottleneck():
    L.hints(
        "bottleneck",
        [
            "A 1×1 convolution is `nn.Conv2d` with a kernel size of 1. How many channels go into `reduce`, and how many come out? And for `expand`?",
            "`self.reduce = nn.Conv2d(channels, middle, 1)`. In `forward`, `x = torch.relu(self.reduce(x))` is the first step; `conv` and `expand` go the same way, and the last line gives back what `expand` and its ReLU make.",
        ],
        why="`reduce` is `nn.Conv2d(channels, middle, 1)` and `expand` is `nn.Conv2d(middle, channels, 1)`, and `forward` sends `x` through `reduce`, `conv` and `expand`, each followed by `torch.relu`. In ResNet-50, each of the three convolutions has batch normalization after it, and the block's input is added to what `expand` gives before the last ReLU, as in [ResNet](/course/open/cnn-resnet?lang=en). Here the block stands in the place of one 3×3 convolution, without that addition.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains the demo's network on the first 6,000 training photos of CIFAR-10, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Choose its block: one 3×3 convolution from 64 to 64, or your `Bottleneck` with 32 or with 16 channels in the middle. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10, and for a run with a bottleneck also exercise 2, and takes a few minutes. While it trains, a chart shows the loss on the training photos every 10 batches, and dashed on the test photos after every epoch. Beside or under it are the network's guesses at that moment for one test photo of each kind, and under them the feature maps its middle makes of a car; both are drawn again as it trains. In the picture after, the gray line is the validation loss of the course's run with one 3×3 convolution and seed 0; when you train that way yourself, its run with 16 in the middle. Then you can rewind to a few epochs of your run, with both pictures as they were then, and see your run next to the course's.
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
    variant_choice = mo.ui.radio(options={"one 3×3 convolution": "plain", "bottleneck, 32 in the middle": "b32", "bottleneck, 16 in the middle": "b16"},
                                 value="bottleneck, 16 in the middle", label="In the middle")
    train_button = mo.ui.run_button(label="Train the network")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_bottleneck(cifar, train_button, variant_choice, result_bottleneck):
    cifar, result_bottleneck  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train the network.*")
    else:
        _variant = variant_choice.value
        _names = {"plain": "one 3×3 convolution", "b32": "bottleneck, 32 in the middle", "b16": "bottleneck, 16 in the middle"}
        _with = {"plain": "one 3×3 convolution in the middle", "b32": "a bottleneck from 64 to 32 channels and back", "b16": "a bottleneck from 64 to 16 channels and back"}
        _now = {}   # the live chart, made at the first report
        _layer = "The middle of the network on a test photo of a car: layer {name}"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your network on CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "The loss of the network so far: on the training photos every 10 batches, and dashed on the 1,000 test photos after every epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "training photos, every 10 batches"), ("loss", True, "test photos, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-bottleneck-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**{_names[_variant].capitalize()}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**{_names[_variant].capitalize()}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant].capitalize()}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `Bottleneck` is in your project, and this cell uses it. Or train with one 3×3 convolution, which needs no exercise.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `Bottleneck` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "bottleneck": "*Your `Bottleneck` in `work/projects/cnn/bottleneck.py` stops with an error, or does not give back 64 maps of 8 by 8. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "cnn": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give what it should. Did you change its file? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _other = "b16" if _variant == "plain" else "plain"
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "one 3×3 convolution", "b16": "16 in the middle",
                       "alt": "The loss of the network after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's run with seed 0 " + ("with a bottleneck from 64 to 16 channels and back." if _other == "b16" else "with one 3×3 convolution, which falls to epoch 6 and then rises.")}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _layer, L.lang, other=_other),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.SHOWN)} photos read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: when is the validation loss lowest, and what happens after?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang, other=_other))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Its validation loss was lowest after epoch **{_best}**." if _best < C.EPOCHS else f" Its validation loss was still falling after epoch {_best}.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""With {_with[_variant]}, the network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters, {C.grouped(_result['block_params'], L.lang)} of them in its block.{_turn} Over 5 seeds, with the course's code: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos, with the lowest validation loss after epoch {_be[0]} to {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Read right after 20 epochs", "x": "read right (%)", "plain": "one 3×3 convolution",
                                        "b32": "bottleneck, 32 in the middle", "b16": "bottleneck, 16 in the middle", "mine": "your run: " + _names[_variant],
                                        "train": "training", "test": "test",
                                        "alt": "Rows of dots between 40 and 100 percent read right: at the top your run, with your training and test photos; under it the course's three runs over 5 seeds, with one 3×3 convolution and with a bottleneck from 64 to 32 and to 16 channels and back."}, L.lang, mine={"variant": _variant, "train": _result["train"], "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Exercise 3. What the bottleneck does here
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "bottleneck-runs",
        question="Over 5 seeds, after 20 epochs. With one 3×3 convolution in the middle, the network has 66,570 parameters, and reads 5,999 to 6,000 of its training photos right and 532 to 572 test photos. With a bottleneck from 64 to 16 channels and back, it has 34,090, and reads 5,077 to 5,510 training photos and 521 to 550 test photos. What do the numbers say?",
        options={
            "worse": "The bottleneck is worse: it reads fewer photos right, of both kinds",
            "better": "The bottleneck is better: it reads more test photos right",
            "right": "With about half the parameters, the bottleneck reads nearly as many test photos right, and learns its training photos less by heart",
        },
        correct="right",
        why={
            "worse": "Fewer training photos, yes, but the network with one 3×3 convolution learns its 6,000 by heart, 5,999 to 6,000 right, and its validation loss ends between 2.81 and 3.12. On the test photos, the ranges overlap: 521 to 550 against 532 to 572.",
            "better": "Not more: 537 on average, against 549, and the ranges overlap. Its lowest validation loss is a little lower, 1.30 to 1.35 against 1.37 to 1.42, and at the end it is 1.56 to 1.86 against 2.81 to 3.12: it is less sure of its wrong guesses, but it gets no more right.",
            "right": "Its block has 4,448 parameters instead of 36,928, 12.0%, and does 278,528 multiplications for a photo instead of 2,359,296. Its training loss ends between 0.30 and 0.49, where the network with one 3×3 convolution ends between 0.006 and 0.018: with fewer parameters, it has less room to learn each photo by heart. On 6,000 photos of 32 by 32, it does nearly as well with far less. GoogLeNet spent such savings on a deeper and wider network; ResNet-50 on four times as many channels between its blocks, and ResNet-101 and ResNet-152 on more blocks.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A 1×1 convolution is the linear layer of [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en), used on every pixel, with the shapes of [Convolution in detail](/course/open/cnn-conv-formal?lang=en): it was in the residual connection of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en), and in a bottleneck it lets the 3×3 convolutions of [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en) work on fewer channels. You counted its parameters as in [Project: your first CNN](/course/open/cnn-pytorch?lang=en), and trained it with the Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) on the photos of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en). Your project `cnn` now has `bottleneck.py`. In the next lesson, *Inception: filters of several sizes side by side*, 1×1 convolutions make the channels fewer before filters of 3 by 3 and 5 by 5 that look at the same maps side by side, as in GoogLeNet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_one_by_one, result_bottleneck, choice_runs):
    L.wrap_up(result_one_by_one, result_bottleneck, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
