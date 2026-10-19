# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-vgg-blocks. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-vgg-blocks", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en) and [LeNet-5, the CNN that read checks](/course/open/cnn-lenet?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_receptive():
    quick_receptive = L.choice(
        "quick-receptive",
        question="Two convolutions of 5 by 5 at stride 1, one after the other, without pooling. How wide is the piece of the image that one number of the second one depends on?",
        options={
            "ten": "10 pixels: twice 5",
            "right": "9 pixels: the second adds 4",
            "five": "5 pixels: every filter is 5 wide",
        },
        correct="right",
        why={
            "ten": "The pieces overlap: the first layer sees 5 pixels, and the second looks at 5 numbers of the first, 1 pixel apart each: 4 pixels more. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "five": "That is the first layer. The second looks at 5 numbers of the first, each with a piece of its own, shifted by one: 4 pixels more. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "right": "5, then 9: the receptive field. A filter of k by k adds k − 1; today you stack filters of 3 by 3, which add 2 each.",
        },
    )
    quick_receptive
    return (quick_receptive,)


@app.cell(hide_code=True)
def quick_check_padding():
    quick_padding = L.choice(
        "quick-padding",
        question="LeNet-5's first convolution keeps 28 by 28 with a filter of 5 by 5 and padding 2. Which padding keeps the size with a filter of 3 by 3?",
        options={
            "two": "2: as in LeNet-5",
            "none": "0: a small filter needs none",
            "right": "1: one ring of zeros",
        },
        correct="right",
        why={
            "two": "With padding 2, a filter of 3 by 3 makes the image 2 wider: 28 + 4 − 3 + 1 = 30. See [LeNet-5, the CNN that read checks](/course/open/cnn-lenet?lang=en).",
            "none": "Without padding, a filter of 3 by 3 fits at 28 − 3 + 1 = 26 places per side. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "right": "Half the filter, rounded down: 2 for 5 by 5, 1 for 3 by 3. Today every convolution has it.",
        },
    )
    quick_padding
    return (quick_padding,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In 2014, Karen Simonyan and Andrew Zisserman of the Visual Geometry Group in Oxford built **VGG**: CNNs of 11 to 19 layers with weights, where your LeNet-5 has 5, with filters of 3 by 3 almost everywhere. On ImageNet, more than a million photos of a thousand kinds, the deeper ones made fewer mistakes, and VGG came second in the classification contest of that year. Its block, a few small convolutions and then pooling, is still in many CNNs. In this step of your project you build that block, and measure what depth does for your own CNN.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working"],
                           2, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_depth():
    guess_depth = L.guess(
        "depth",
        question="You build networks of two VGG blocks, as your SmallCNN, with 1, 2, 4 or 8 convolutions of 3 by 3 per block, and train each for 5 epochs with your Trainer on the same 6,000 images. Which reads the most of the 1,000 test images right?",
        options={
            "eight": "8 per block: the deepest sees the most",
            "four": "4 per block: deep, but not too deep",
            "right": "1 or 2 per block: small images need few layers",
        },
        reveal="**1 or 2 per block.** Over 5 seeds, with the course's code, 1 per block, your SmallCNN, reads 864 to 879 of the 1,000 right, and 2 per block 859 to 883. With 4 per block it is 832 to 869, below 1 per block with every seed. With 8 per block, 17 layers with weights, not one run learns anything: 105 right, one kind for every image. On 6,000 small images, depth does not help here; and a plain network that is too deep does not train at all.",
    )
    guess_depth
    return (guess_depth,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Small filters see as far.** As in [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), every convolution of 3 by 3 adds 2 to the width of the piece a number depends on: two of them see 5 by 5, three see 7 by 7, as one filter of 7 by 7 would.

        **With fewer weights.** From $C$ to $C$ channels, one convolution of 7 by 7 has $49C^2$ weights: $C$ filters of $C \times 7 \times 7$. Three of 3 by 3 have $3 \times 9C^2 = 27C^2$. The single convolution needs 81% more. And the stack has a ReLU after every convolution: three where the one convolution has one.

        **A VGG block** is $n$ convolutions of 3 by 3 with padding 1, each with a ReLU after it, and then max pooling of 2 by 2. The convolutions keep the height and width; the pooling halves them. VGG doubles the channels from block to block: 64, 128, 256, 512, and 512 in the last.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **VGG block** | V G G block | $n$ convolutions of 3 by 3 with padding 1, each with a ReLU, then max pooling of 2 by 2 | `VGGBlock(c_in, c_out, n)` |
        | **VGG** | V G G | the CNNs of Simonyan and Zisserman (2014): five VGG blocks and three linear layers, 11 to 19 layers with weights in all | |
        """),
        L.watch_out("Deeper is harder to train. VGG gave its deeper networks the first and last layers of a shallower one that had learned, because bad random starting values can stall a deep network. PyTorch's own starting values for `nn.Conv2d` are smaller than He initialization asks: the differences between images shrink in every layer, and after 16 convolutions every image gives the same logits. With the He initialization of [Initialization](/course/open/nn-initialization?lang=en), 8 per block does learn: 847 to 858 right over 3 seeds, and still not more than with 1."),
        mo.md(r"""
        The picture: the weights from $C$ to $C$ channels, in units of $C^2$, against how wide a piece they see. One convolution of $k$ by $k$ grows with the square of $k$; a stack of 3 by 3 grows by 9 for every 2 pixels more.
        """),
        mo.Html(C.weights_picture({"title": "Weights for the same view", "x": "width of the piece seen, in pixels", "y": "weights, in C²",
                           "one": "one convolution of k by k", "stack": "a stack of 3 by 3",
                           "alt": "Two lines of weights in units of C squared against the width of the piece seen, 3 to 11 pixels. One convolution of k by k, dashed, rises from 9 through 25 and 49 to 121. A stack of 3 by 3 rises from 9 through 18 and 27 to 45. At 7 pixels: 49 against 27."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A VGG block with $n$ convolutions, from $C_\text{in}$ to $C_\text{out}$ channels:

    1. $n$ convolutions of 3 by 3 with padding 1, each with a ReLU after it: the first from $C_\text{in}$ to $C_\text{out}$ channels, the others from $C_\text{out}$ to $C_\text{out}$. The height and width stay.
    2. Max pooling of 2 by 2 at the end: the height and width halve.
    3. The next block starts from the $C_\text{out}$ channels of this one, and usually has twice as many.
    4. After the last block: one row per image, and the output head.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 of the recipe, on an image of 11 by 11 that is 0 everywhere but a 1 in the middle, with a convolution whose weights are all 1, used three times. How many numbers does the 1 reach? The comments with `#` say which step each part is. The third convolution is not done yet: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_reach():
    import torch
    from torch import nn

    _x = torch.zeros(1, 1, 11, 11)
    _x[0, 0, 5, 5] = 1.0
    # 1. convolutions of 3 by 3 with padding 1: here all weights 1 and the bias 0
    _conv = nn.Conv2d(1, 1, 3, padding=1)
    nn.init.ones_(_conv.weight)
    nn.init.zeros_(_conv.bias)
    _h = _conv(_x)
    print(_h.shape, (_h != 0).sum().item())
    _h = _conv(_h)
    print(_h.shape, (_h != 0).sum().item())
    _h = _h  # complete: _conv(_h)
    print(_h.shape, (_h != 0).sum().item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The size stays 11 by 11 with padding 1. After one convolution, the 1 reaches 9 numbers, a piece of 3 by 3; after two, 25, a piece of 5 by 5. As it is, the third line says 25 again. Completed, it says 49: 7 by 7. Turned around, one number after three convolutions depends on a piece of 7 by 7 of the image, as one filter of 7 by 7 would. Here that is one convolution, used three times; three convolutions of 3 by 3 of their own have 27 weights, one of 7 by 7 has 49.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Three blocks
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "vgg-shape",
        question="A network for images of 28 by 28 has three VGG blocks, with 16, 32 and 64 channels. How many numbers per image go into its output head?",
        options={
            "keep": "64 × 7 × 7 = 3,136: the convolutions keep the size",
            "right": "64 × 3 × 3 = 576: three halvings, the last rounded down",
            "up": "64 × 4 × 4 = 1,024: 7 halves to 3.5, rounded up",
        },
        correct="right",
        why={
            "keep": "The convolutions keep it, but every block ends with pooling: three halvings, not two.",
            "up": "Max pooling of 2 by 2 drops a last row and column that do not fill a block: 7 becomes 3, rounded down.",
            "right": "28, 14, 7, and then 3: the last pooling drops the seventh row and column. 64 channels of 3 by 3 is 576.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_params():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Small filters against one large one

        Write `stack_vs_one(c, n)`: it gives two numbers of parameters, weights and bias terms together. First that of a stack of `n` convolutions of 3 by 3 from `c` to `c` channels; then that of one convolution from `c` to `c` channels that sees the same piece, $2n + 1$ wide. The check calls `stack_vs_one(64, 3)`, which should give `(110784, 200768)`, and other numbers of channels and convolutions.
        """),
        L.task("Make `stack_vs_one` count the parameters of a stack of small convolutions and of one large convolution that sees as far."),
    ])
    return


@app.cell(hide_code=True)
def exercise_params():
    box_params = L.code_box("stack-params", lines=8)
    box_params
    return (box_params,)


@app.cell(hide_code=True)
def check_params(box_params):
    result_params = L.check("stack-params", box_params.value)
    result_params
    return (result_params,)


@app.cell(hide_code=True)
def hints_params():
    L.hints(
        "stack-params",
        [
            "A convolution from `c` to `c` channels with a filter of `k` by `k` has `c` filters of `c` by `k` by `k`, and `c` bias terms.",
            "The stack is `n` times a convolution with `k = 3`. The one convolution has `k = 2 * n + 1`.",
        ],
        why="The one convolution grows with the square of its width, the stack only with its length: for 3 convolutions at 64 channels, 200,768 against 110,784, 81% more.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The VGG block

        Complete `VGGBlock(c_in, c_out, n)`: in `__init__` its `n` convolutions, in an `nn.ModuleList` as in [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en), and its pooling; in `forward` steps 1 and 2 of the recipe. The check makes blocks after `torch.manual_seed(0)` and compares what comes out, so make the convolutions in the order in which the data passes them, with PyTorch's own starting values; it also counts the parameters PyTorch sees. When it is right, the check saves it in your project, as `work/projects/cnn/vgg.py`.
        """),
        L.task("Make `VGGBlock` run `n` convolutions of 3 by 3, each with a ReLU, and then max pooling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("vgg-block", lines=16)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("vgg-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "vgg-block",
        [
            "In `__init__`, fill an `nn.ModuleList` in a `for` loop that runs `n` times. Only the first convolution gets `c_in` channels; every next one gets the `c_out` of the one before.",
            "In `forward`, go over `self.convs` with a `for` loop and put every convolution inside `torch.relu`; after the loop, the pooling.",
        ],
        why="In an `nn.ModuleList`, PyTorch finds the convolutions: `model.parameters()` gives their weights, and the Trainer trains them. In a plain Python list, the check would count 0 parameters.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. With your `VGGBlock`, it builds networks of two blocks, from 1 to {C.WIDTHS[0]} channels and from {C.WIDTHS[0]} to {C.WIDTHS[1]}, as your SmallCNN, with {", ".join(str(n) for n in C.DEPTHS[:-1])} and {C.DEPTHS[-1]} convolutions per block. With 1 per block, it is your SmallCNN, with the same starting values. It trains each with your Trainer: {C.EPOCHS} epochs on the 6,000 training images, a learning rate of {C.number(C.LR, 2, L.lang)} with a warmup of one epoch, one run each, with seed 0, and counts the parameters with your `count_params`. Where a module of yours is missing or broken, the course's stands in. It works once exercise 3 is done, and takes one to two minutes.
    """)
    return


@app.cell(hide_code=True)
def depth_button_cell():
    depth_button = mo.ui.run_button(label="Train the four networks")
    depth_button
    return (depth_button,)


@app.cell(hide_code=True)
def demo_depth(depth_button, result_block):
    result_block  # run again when exercise 3 saves your module
    if not depth_button.value:
        _out = mo.md("*Press the button to train the four networks.*")
    else:
        _result = C.demo()
        if _result is None:
            _out = mo.md("*Finish exercise 3 first: then your `VGGBlock` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "slow": "*Your project takes too long here: more than five minutes to train one network, on a computer that is not busy with other work. Does your `VGGBlock` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "vgg": "*Your `VGGBlock` in `work/projects/cnn/vgg.py` stops with an error, while the networks are made. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result])
        else:
            _rows = "\n".join(f"| {n} | {2 * n + 1} | {C.grouped(p, L.lang)} | {right} |" for n, p, right in _result["rows"])
            _who = "" if not _result["stand_ins"] else "\n\nThe course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _out = mo.md(f"""| convolutions per block | layers with weights | parameters | right |
|---|---|---|---|
{_rows}

Over 5 seeds, with the course's code: 864 to 879 right with 1 per block, 859 to 883 with 2, 832 to 869 with 4, and 105 in every run with 8: one kind for every image.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The receptive field of [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), now as the reason to stack small filters: they see as far as one large filter, with fewer weights and more ReLUs. Your project `cnn` now has `vgg.py`. Deeper was not better here: on 6,000 small images of 28 by 28, one or two convolutions per block did best, and with eight per block, training got stuck. The next extra lesson, *ResNet: depth that keeps working*, gives a deep CNN the residual connections of your deep MLP in [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_params, result_block):
    L.wrap_up(choice_shape, result_params, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
