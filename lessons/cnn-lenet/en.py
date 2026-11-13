# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-lenet. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-lenet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: your first CNN](/course/open/cnn-pytorch?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="How many parameters does `nn.Conv2d(3, 4, 5)` have?",
        options={
            "bias": "300: 4 filters of 3 × 5 × 5, without bias terms",
            "channels": "104: 4 filters of 5 × 5, one per output channel, and their bias terms",
            "right": "304: 4 filters of 3 × 5 × 5, and a bias term each",
        },
        correct="right",
        why={
            "bias": "Every filter also has a bias term: 4 more. See [Project: your first CNN](/course/open/cnn-pytorch?lang=en).",
            "channels": "A filter covers every input channel: 3 × 5 × 5 = 75 weights, not 25. See [Project: your first CNN](/course/open/cnn-pytorch?lang=en).",
            "right": "4 × 3 × 5 × 5 = 300 weights, and 4 bias terms. Today you count a whole network of 1998 that way.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="The feature maps of a mini-batch have shape (4, 16, 3, 3). What shape does `x.reshape(len(x), -1)` give?",
        options={
            "right": "(4, 144): one row per image",
            "all": "(576,): one row for the whole mini-batch",
            "channels": "(4, 16, 9): one row per channel",
        },
        correct="right",
        why={
            "all": "That is `x.reshape(-1)`. `len(x)` keeps the 4 images apart. See [Project: your first CNN](/course/open/cnn-pytorch?lang=en).",
            "channels": "`-1` takes everything after the first axis together: 16 × 3 × 3 = 144 numbers per image. See [Project: your first CNN](/course/open/cnn-pytorch?lang=en).",
            "right": "16 × 3 × 3 = 144 numbers per image, for the linear layers after it. LeNet-5 does the same.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your SmallCNN has two blocks of convolution and pooling, and then an output head. That shape is old. In 1998, Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner described **LeNet-5**: a CNN of two such blocks and three linear layers that read handwritten digits. A system built on it read several million bank checks a day. Later CNNs, such as VGG and ResNet of the next two lessons, are the same blocks, stacked deeper, with a few new tricks. In this step of your project you rebuild LeNet-5 in PyTorch, and train it against your SmallCNN.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working"],
                           1, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_better():
    guess_better = L.guess(
        "better",
        question="LeNet-5 has 61,706 parameters, almost 7 times the 9,098 of your SmallCNN. Both train for 5 epochs with your Trainer, on the same 6,000 images. Which reads more of the 1,000 test images right?",
        options={
            "lenet": "LeNet-5",
            "right": "Your SmallCNN",
            "same": "Both about as many",
        },
        reveal="**Your SmallCNN.** Over 5 seeds, with the course's code, LeNet-5 reads 828 to 852 of the 1,000 right, and SmallCNN 864 to 879: more every time, with a seventh of the parameters. LeNet-5 has the tanh and the average pooling of its time; with ReLU and max pooling, which took over later, the same network reads 845 to 866: closer, and still below SmallCNN with every seed. More parameters alone do not make a better model.",
    )
    guess_better
    return (guess_better,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        LeNet-5 has the shape of your SmallCNN, with other sizes and two older choices.

        **Two blocks.** Block 1 is a convolution of 5 by 5 from 1 to 6 channels, block 2 one of 5 by 5 from 6 to 16 channels; each is followed by tanh and pooling of 2 by 2. The paper gave LeNet-5 images of 32 by 32; the 28 by 28 of Fashion-MNIST with padding 2 comes to the same. Block 2 has no padding, so its feature maps shrink from 14 by 14 to 10 by 10.

        **Three linear layers.** One row of numbers per image goes into `fc1`, to 120 numbers, then `fc2`, to 84, each with tanh after it, and then the output head, to the 10 logits.

        **tanh**, where your SmallCNN has ReLU: the S-shaped curve from −1 to 1 of [Initialization](/course/open/nn-initialization?lang=en). Numbers near 0 change little, and large ones end up near −1 or 1. **Average pooling**, where your SmallCNN has max pooling: of each block of 2 by 2 it keeps the mean, not the largest number. Of 1, 2, 3 and 6 it keeps 3, where max pooling keeps 6.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **LeNet-5** | | the CNN of LeCun and others (1998): two blocks of convolution, tanh and pooling, and three linear layers | |
        | **tanh** | tan h | a function that squeezes every number between −1 and 1, with 0 staying 0 | `torch.tanh(x)` |
        | `nn.AvgPool2d` | n n average pool 2 d | average pooling of every channel, in blocks of $k$ by $k$ | `nn.AvgPool2d(2)` |
        """),
        L.watch_out("The LeNet-5 of the paper differs in details from the one you build, and from most rebuilds: its pooling had a weight and a bias term of its own per channel, not every feature map of block 2 looked at all 6 of block 1, its tanh was scaled, and its output layer measured distances to fixed patterns instead of giving logits. The blocks, the sizes and the number of parameters, about 60,000, are the same."),
        mo.md(r"""
        The picture: one image on its way through LeNet-5, with its shape after every step. Each stack has a square for every channel, at most 6 drawn, as big as its height and width; a row of numbers has a box for every number, at most 10 drawn, with dots for the rest. In a mini-batch, every shape gets $B$ in front.
        """),
        C.shapes_picture({"title": "One image through LeNet-5", "input": "one gray image", "maps": "{0} feature maps of {1} by {1}",
                          "fc": "{0} numbers", "logits": "10 logits, one per kind",
                          "conv1": "conv 5×5 with padding 2, tanh", "conv2": "conv 5×5 without padding, tanh",
                          "pool": "average pooling 2 by 2", "flat": "one row of 400, fc1, tanh", "linear": "fc2, tanh",
                          "head": "the output head",
                          "alt": "A gray image of 1 by 28 by 28 becomes 6 feature maps of 28 by 28 after a convolution of 5 by 5 with padding 2 and tanh, 6 of 14 by 14 after average pooling, 16 of 10 by 10 after a convolution without padding and tanh, and 16 of 5 by 5 after pooling. One row of 400 numbers becomes 120 numbers after fc1 and tanh, 84 after fc2 and tanh, and 10 logits after the output head."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    LeNet-5 for images of 28 by 28 with one channel:

    1. Block 1: a convolution of 5 by 5 from 1 to 6 channels with padding 2, tanh, and average pooling of 2 by 2: $(B, 6, 14, 14)$.
    2. Block 2: a convolution of 5 by 5 from 6 to 16 channels without padding, tanh, and average pooling: $(B, 16, 5, 5)$.
    3. One row per image: `x.reshape(len(x), -1)`, 16 × 5 × 5 = 400 numbers.
    4. `fc1` from 400 to 120 and `fc2` from 120 to 84, each with tanh after it, and the output head from 84 to 10: the logits.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    First tanh on five numbers, then step 1 of the recipe on the first 2 training images. The comments with `#` say which part is which. The pooling is not done yet: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_block():
    import torch
    from torch import nn

    # tanh: every number between -1 and 1
    print(torch.tanh(torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])))
    _X, _y, _, _ = C.fashion()
    _x = _X[:2]
    # 1. block 1: convolution with padding 2, tanh, average pooling
    torch.manual_seed(0)
    _conv = nn.Conv2d(1, 6, 5, padding=2)
    _pool = nn.AvgPool2d(2)
    _h = torch.tanh(_conv(_x))
    print(_h.shape)
    _h = _h  # complete: _pool(_h)
    print(_h.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    tanh makes 0.5 into 0.4621, a little smaller, and 3 into 0.9951, almost 1; negative numbers go the same way, below 0. With padding 2, the convolution keeps the 28 by 28: (2, 6, 28, 28). As it is, the second shape is that one again. Completed, the average pooling halves the height and width: (2, 6, 14, 14), what block 2 gets.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Another size of image
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "lenet-shape",
        question="You give your LeNet-5 gray images of 32 by 32 instead of 28 by 28, with the same layers and padding 2 in block 1. What must `fc1` become?",
        options={
            "right": "`nn.Linear(16 * 6 * 6, 120)`: 6 by 6 after block 2",
            "same": "`nn.Linear(16 * 5 * 5, 120)`: the layers before it do not depend on the size of the image",
            "quarter": "`nn.Linear(16 * 8 * 8, 120)`: two poolings make it a quarter, 32 / 4",
        },
        correct="right",
        why={
            "same": "Their weights do not depend on it, but their output does: 32 stays 32 in block 1 and becomes 16 after pooling, 12 after block 2's convolution, and 6 after pooling.",
            "quarter": "The poolings halve twice, but block 2's convolution, without padding, first takes 4 off: 16 − 4 = 12, then 6.",
            "right": "32, then 16 after pooling, 12 after block 2's convolution, 6 after pooling: 16 × 6 × 6 = 576 numbers per image. The convolutions work on any size; only `fc1` needs to know it.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_lenet():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. LeNet-5 in PyTorch

        Complete `LeNet5`: in `__init__` its layers, and in `forward` steps 1 to 4 of the recipe, for a mini-batch `x` of shape $(B, 1, 28, 28)$; it gives the logits, of shape $(B, 10)$. Make the layers with weights in the order in which the data passes them: `conv1`, `conv2`, `fc1`, `fc2`, `out`. The check makes the network after `torch.manual_seed(0)` and compares its logits for 2 and for 3 random images, and once more with all its parameters 5 times as large. When it is right, the check saves it in your project, as `work/projects/cnn/lenet.py`.
        """),
        L.task("Make `LeNet5` give 10 logits per image, from two blocks of convolution, tanh and average pooling, and three linear layers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lenet():
    box_lenet = L.code_box("lenet5", lines=22)
    box_lenet
    return (box_lenet,)


@app.cell(hide_code=True)
def check_lenet(box_lenet):
    result_lenet = L.check("lenet5", box_lenet.value)
    result_lenet
    return (result_lenet,)


@app.cell(hide_code=True)
def hints_lenet():
    L.hints(
        "lenet5",
        [
            "In `__init__`, six attributes of `self`: `conv1` and `conv2` as `nn.Conv2d`, one `nn.AvgPool2d(2)` that both blocks share, and `fc1`, `fc2` and `out` as `nn.Linear`. The comment in the box gives their sizes.",
            "In `forward`, each block is one line: the convolution, `torch.tanh` around it, and the pooling around that: the two lines of the worked example in one. After the two blocks, one row per image as in your SmallCNN; then `fc1` and `fc2`, each inside `torch.tanh`, and the output head without.",
        ],
        why="tanh comes after every layer but the output head: the logits go to the cross-entropy loss, which does the softmax itself. Without tanh between them, `fc1` and `fc2` together would be one linear layer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Exercise 3. Where the parameters are
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where-params",
        question="Where are most of the 61,706 parameters of LeNet-5?",
        options={
            "conv": "In `conv1` and `conv2`: a filter for every channel",
            "head": "In `out`: a weight for every kind and every number of `fc2`",
            "right": "In `fc1`: almost 4 of every 5, a weight for every number of the row and every neuron",
        },
        correct="right",
        why={
            "conv": "`conv1` has 6 × 25 + 6 = 156 and `conv2` 16 × 6 × 25 + 16 = 2,416: a filter of 5 by 5 has 25 weights per input channel, and serves every place of the image.",
            "head": "`out` has 84 × 10 + 10 = 850: a small layer at the end.",
            "right": "`fc1` has 400 × 120 + 120 = 48,120: a weight for every number of the row of 400 and every one of its 120 neurons, as in an MLP. The two convolutions together have 2,572.",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The cell below uses your project. It trains your `SmallCNN` of [Project: your first CNN](/course/open/cnn-pytorch?lang=en), and then your `LeNet5`, with your Trainer: {C.EPOCHS} epochs on the 6,000 training images, a learning rate of {C.number(C.LR, 2, L.lang)} with a warmup of one epoch, the same for both, one run each, with seed 0. It counts their parameters with your `count_params`. Where a module of yours is missing or broken, the course's stands in. It works once exercise 2 is done, and takes about 10 seconds. While each network trains, a chart shows its loss on the training images every 10 batches, and dashed on the test images after every epoch: first of your SmallCNN, then of your LeNet-5, with the validation loss of SmallCNN thin and dashed beside it. Beside or under it are the network's guesses at that moment for one test image of each kind, and under them the feature maps of its second convolution, `conv2`, on an ankle boot; both are drawn again as it trains. Then you can rewind through both runs, epoch by epoch, with the pictures of both networks as they were then.
    """)
    return


@app.cell(hide_code=True)
def demo_lenet(result_lenet):
    result_lenet  # run again when exercise 2 saves your module
    _now = {}   # the live chart of each network, made at its first report, and its validation loss so far
    _heads = {"small": {"guesses": "Your SmallCNN: its guess for one test image of each kind",
                        "layer": "Its second convolution on a test image of an ankle boot: layer {name}"},
              "lenet": {"guesses": "Your LeNet-5: its guess for one test image of each kind",
                        "layer": "Its second convolution on a test image of an ankle boot: layer {name}"}}

    def _progress(_p):
        _run = _p["run"]
        if _run not in _now:
            _le = _run == "lenet"
            _now[_run] = L.live({"title": "Your LeNet-5" if _le else "Your SmallCNN", "x": "epoch", "y": "loss",
                                 "alt": "The loss of your LeNet-5 so far: on the training images every 10 batches, and dashed on the 1,000 test images after every epoch; thin and dashed, the validation loss of your SmallCNN." if _le else "The loss of your SmallCNN so far: on the training images every 10 batches, and dashed on the 1,000 test images after every epoch."}, _p["epochs"], low=0.0, window=5,
                                key=[("loss", False, "training images, every 10 batches"), ("loss", True, "test images, after every epoch")])
            _now[_run].watch(lambda: C.now_pictures(_p["look"](), _heads[_run], L.lang, f"nv-lenet-{_run}-now"))
            _now[_run + "-test"] = []
            if _le and "small-test" in _now:   # the validation loss of SmallCNN, thin and dashed beside LeNet-5
                _now["lenet"].reference(_now["small-test"], "SmallCNN", role="ink-2")
        _who = "Your LeNet-5" if _run == "lenet" else "Your SmallCNN"
        if "val_loss" in _p:   # the test images after every epoch, dashed
            _now[_run + "-test"].append((_p["epoch"], _p["val_loss"]))
            _now[_run].reference(_now[_run + "-test"], "", role="loss", replace=True)
            _now[_run].line(f"**{_who}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
        elif "loss" in _p:   # the training images every 10 batches
            _now[_run].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                           f"**{_who}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
        else:
            _now[_run].line(f"**{_who}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

    _result = C.demo(_progress)
    if _result is None:
        _out = mo.md("*Finish exercise 2 first: then your `LeNet5` is in your project, and this cell uses it.*")
    elif isinstance(_result, str):
        _note = {
            "slow": "*Your project takes too long here: more than two minutes to train one network. Does your `LeNet5` or your Trainer do more work than it needs, such as a loop that runs too often?*",
            "lenet": "*Your `LeNet5` in `work/projects/cnn/lenet.py` stops with an error, while it is made, counted or trained. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
            "trainer": "*Your Trainer in `work/projects/nn/trainer.py`, your `SmallCNN` in `work/projects/cnn/small_cnn.py` or your `count_params` in `work/projects/cnn/count.py` stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
        }[_result]
        _last = _now.get("lenet") or _now.get("small")   # a run that stops keeps its chart so far
        _out = _last.stopped(_note) if _last else mo.md(_note)
    else:
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
        _labels = {"title": "Validation loss after every epoch", "x": "epoch", "y": "validation loss", "small": "SmallCNN", "alt": "The validation loss of your LeNet-5 after every epoch, and thin and dashed that of your SmallCNN."}
        _curves = _result.get("curves", {})
        _when = list(range(1, len(_curves.get("lenet", [])) + 1))
        if any(len(_result["looked"][_n]) != len(_curves.get(_n, [])) or not all(_result["looked"][_n]) for _n in C.RUNS):   # an epoch it could not look at
            _when = []

        def _frame(_e):
            _e = int(_e)
            _right = {_n: sum(_g == _k for _g, _k in zip(_result["looked"][_n][_e - 1], _result["truth"])) for _n in C.RUNS}
            return (C.epoch_frame(_result, _e, _labels, _heads, L.lang),
                    f"After epoch {_e}: of these 10 test images, your SmallCNN reads {_right['small']} right, and your LeNet-5 {_right['lenet']}.")

        if len(_when) > 1:   # rewinding through both runs, with both in the picture
            _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                              task="Go back to epoch 1, then forward again: where do the guesses and the feature maps of your two networks differ?")]
        else:
            _pic = [mo.Html(C.curves_picture(_curves, _labels, L.lang))] if _curves else []
        _out = mo.vstack(_pic + [mo.md(f"""Of the 1,000 test images, your LeNet-5 reads **{_result['lenet']}** right, with {C.grouped(_result['params_lenet'], L.lang)} parameters, counted with your `count_params`; your SmallCNN reads **{_result['small']}** right, with {C.grouped(_result['params_small'], L.lang)}. Over 5 seeds, with the course's code, LeNet-5 reads 828 to 852 right and SmallCNN 864 to 879. The chart shows the validation loss of both after every epoch, SmallCNN's dashed.{_who}""")])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The blocks of your SmallCNN of [Project: your first CNN](/course/open/cnn-pytorch?lang=en), as LeCun and others put them together in 1998, with the tanh of their time and the average pooling of [Channels and pooling](/course/open/cnn-channels-pooling?lang=en). Most of its parameters sit in a linear layer, as in an MLP; the convolutions need few, because a filter is shared over the whole image. Your project `cnn` now has `lenet.py` next to `small_cnn.py`. The next lesson, *VGG: deeper with small filters*, stacks many convolutions of 3 by 3 into a deeper network.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_lenet, choice_where):
    L.wrap_up(choice_shape, result_lenet, choice_where)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
