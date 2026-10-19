# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-conv-formal. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-conv-formal", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_depends():
    quick_depends = L.choice(
        "quick-depends",
        question="Which pixels of the image does one number of a feature map depend on?",
        options={
            "all": "All pixels: the filter slides over the whole image",
            "right": "Only the pixels of the piece under the filter at that place",
            "row": "The pixels of its own row of the image, from left to right",
        },
        correct="right",
        why={
            "all": "The filter slides over the whole image, but each number of the feature map is the sum for one place only. See [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).",
            "row": "The piece under the filter is k by k: rows and columns around the place. See [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).",
            "right": "That is locality: each number looks at a small piece. Today you see how that piece grows when layers are stacked.",
        },
    )
    quick_depends
    return (quick_depends,)


@app.cell(hide_code=True)
def quick_check_edge():
    quick_edge = L.choice(
        "quick-edge",
        question="In `convolve`, the filter goes only to places where it fits completely. What does that mean for the pixels at the edge of the image?",
        options={
            "right": "They are under the filter less often than pixels in the middle",
            "skip": "No place of the filter covers them, so they are skipped",
            "same": "Nothing: every pixel, at the edge or in the middle, is under the filter equally often",
        },
        correct="right",
        why={
            "skip": "The corner pixel is under the filter at one place, the first one; it is not skipped, but it counts only once. See [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).",
            "same": "A pixel in the middle is under k × k places of the filter, a corner pixel under one. See [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).",
            "right": "And the feature map is smaller than the image. Today, padding lets the filter also sit over the edge: edge pixels count more often, and the feature map can stay as big as the image.",
        },
    )
    quick_edge
    return (quick_edge,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A real convolutional layer has three settings besides its weights: how big the filter is, how far it moves at a time, and what it does at the edge. Together they decide how big the output is, and so how many numbers the next layer gets and how much it costs to compute. They also decide how much of the image a number deep in the network can see: an object only gets recognized when the network looks at enough of it at once. Every CNN you build or read, from LeNet to ResNet, is described in these settings.
    """)
    return


@app.cell(hide_code=True)
def predict_field():
    guess_field = L.guess(
        "three-layers",
        question="Three convolutional layers with filters of 3 by 3, one on top of the other, each filter moving one place at a time. How many pixels wide is the piece of the image that one number of the third layer's output depends on?",
        options={
            "three": "3: the size of one filter",
            "right": "7: each layer adds 2",
            "nine": "9: three filters of 3",
        },
        reveal="**7 by 7.** A number of the first layer sees 3 by 3 pixels. A number of the second sees 3 by 3 numbers of the first, each of which sees its own 3 by 3 pixels, shifted by one: together 5 by 5. The third adds 2 again: 7 by 7. Small filters, stacked, look at a large part of the image.",
    )
    guess_field
    return (guess_field,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **kernel size** $k$ is the size of the filter, $k$ by $k$: *kernel* is another word for filter, and the course keeps it only in this name. The **stride** $s$ is how many places the filter moves at a time: with $s = 2$ it skips every other place, and the output is about half as wide. **Padding** $p$ puts a frame of zeros, $p$ wide, around the image, so that the filter also fits at the edge; with $k = 3$, $p = 1$ and $s = 1$, the output is as big as the image. Together they give the size of the output of an image of $n$ by $n$:
        """),
        L.formula(r"n_\text{out} = \left\lfloor \frac{n + 2[[hyper:p]] - [[hyper:k]]}{[[hyper:s]]} \right\rfloor + 1"),
        mo.md(r"""
        $n + 2p$ is the width of the padded image, $n + 2p - k$ the number of places the filter can move to the right of its first place, divided by $s$ the number of steps it takes, and $+1$ counts the first place. The brackets $\lfloor \, \rfloor$ round down: a place where the filter would stick out does not count. In Python that is `//`, division rounded down: `7 // 2` gives `3`.

        Each number of the output depends on a $k$ by $k$ piece of the input. Stack layers, and a number of the second layer depends on $k$ by $k$ numbers of the first, each of which sees its own piece: together a wider piece. That piece of the image is the **receptive field**. With filters of 3 by 3 at stride 1, every layer adds 2 to its width: 3, 5, 7. A stride makes what the layers after it add larger: after a layer with stride 2, neighboring numbers are 2 pixels apart, so every 3 by 3 layer after it adds 4 instead of 2; each further stride of 2 doubles that again.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **kernel size** | | the size of the filter, $k$ by $k$ | `k` |
        | **stride** | | how many places the filter moves at a time | `s` |
        | **padding** | | a frame of zeros around the input, so the filter also fits at the edge | `p` |
        | **receptive field** | | the piece of the image one number of a layer depends on | |
        | $\lfloor x \rfloor$ | floor of x | $x$ rounded down to a whole number | `//` |
        """),
        L.watch_out("With an even kernel size, padding cannot be the same on both sides and keep the size at stride 1: $k = 3$ with $p = 1$ and $k = 5$ with $p = 2$ keep it, which is one reason filters are almost always odd. And the zeros of the padding are not part of the picture: around a bright image, the filter for vertical edges sees an edge all along the left and right border."),
        mo.md(r"""
        The picture: an image of 5 by 5 with a frame of zeros, 1 wide, a filter of 3 by 3 and a stride of 2. The solid frame is the filter at its first place, the dashed one at its second, 2 places on; the first two boxes of the output are the numbers of those two places. The output is $\lfloor (5 + 2 - 3) / 2 \rfloor + 1 = 3$ by 3.
        """),
        C.conv_picture({"input": "input, with padding 1", "output": "output", "caption": "n = 5, k = 3, s = 2, p = 1: an output of 3 by 3",
                        "alt": "A grid of 7 by 7 boxes: an image of 5 by 5 with a frame of zeros. A solid frame of 3 by 3 covers the top left corner, a dashed one starts 2 boxes further to the right. Next to it, an output of 3 by 3, of which the first two boxes are colored."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A convolution with kernel size $k$, stride $s$ and padding $p$:

    1. Pad: an array of zeros of $n + 2p$ by $n + 2p$, with the image in the middle.
    2. Compute the output size, $\lfloor (n + 2p - k)/s \rfloor + 1$: in Python, `//` rounds down.
    3. For every output place `r`, `c`: the $k$ by $k$ piece of the padded image that starts at row `r * s` and column `c * s`, times the filter, added up.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An image of 4 by 4 with padding 1, a filter of 3 by 3 ones, which adds up the piece under it, and stride 2. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_stride():
    import numpy as np

    _img = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])
    _K = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    _n, _k, _s, _p = 4, 3, 2, 1
    # 1. pad: a frame of zeros
    _x = np.zeros((_n + 2 * _p, _n + 2 * _p))
    _x[_p:_p + _n, _p:_p + _n] = _img
    # 2. the output size
    _m = (_n + 2 * _p - _k) // _s + 1
    print(_m)
    # 3. the filter at every place, _s places apart
    _out = np.zeros((_m, _m))
    for _r in range(_m):
        for _c in range(_m):
            _out[_r, _c] = (_x[_r:_r + _k, _c:_c + _k] * _K).sum()  # complete: start at row _r * _s and column _c * _s
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The output size is $\lfloor (4 + 2 - 3) / 2 \rfloor + 1 = 2$. As it is, the filter moves one place at a time: it sums the pieces at rows and columns 0 and 1, `[[4, 9], [7, 13]]`, the top left corner of a stride-1 output. Completed, with `_x[_r * _s:_r * _s + _k, _c * _s:_c * _s + _k]`, the pieces start at 0 and 2: `[[4, 11], [9, 13]]`. The first number, 4, is the sum of the top left piece: five zeros of the frame and 1 + 2 + 0 + 1 of the image.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_receptive():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A stride in the stack
    """)
    return


@app.cell(hide_code=True)
def choice_receptive_cell():
    choice_receptive = L.choice(
        "receptive",
        question="Two layers: first a filter of 3 by 3 with stride 2, then a filter of 3 by 3 with stride 1. How many pixels wide is the piece of the image that one number of the second layer's output depends on?",
        options={
            "five": "5: each layer adds 2, as without a stride",
            "nine": "9: three numbers of the first layer, each 3 pixels wide",
            "right": "7: the second layer adds twice as much after the stride",
        },
        correct="right",
        why={
            "five": "After the stride of 2, neighboring numbers of the first layer are 2 pixels apart, so the second layer's 2 extra numbers cover 4 extra pixels.",
            "nine": "The three numbers of the first layer overlap: each sees 3 pixels, and they start 2 pixels apart, so together they see 2 + 2 + 3 = 7.",
            "right": "The first layer sees 3 pixels; the second adds 2 numbers of the first, 2 pixels apart each: 3 + 2 × 2 = 7.",
        },
    )
    choice_receptive
    return (choice_receptive,)


@app.cell(hide_code=True)
def text_exercise_out_size():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The output size

        Complete `out_size(n, k, s, p)`: the width of the output of an image of `n` by `n`, with kernel size `k`, stride `s` and padding `p`, as a whole number. Replace the `...` with the formula of The idea, in Python. The check calls `out_size(28, 5, 2, 0)`, which should give `12`, and tries other sizes.
        """),
        L.task("Make `out_size` give the size of the output of a convolution."),
    ])
    return


@app.cell(hide_code=True)
def exercise_out_size():
    box_out_size = L.code_box("out-size", lines=3)
    box_out_size
    return (box_out_size,)


@app.cell(hide_code=True)
def check_out_size(box_out_size):
    result_out_size = L.check("out-size", box_out_size.value)
    result_out_size
    return (result_out_size,)


@app.cell(hide_code=True)
def hints_out_size():
    L.hints(
        "out-size",
        [
            "The width of the padded image is `n + 2 * p`; the filter can move `n + 2 * p - k` places to the right of its first place.",
            "Divide that by `s` and round down with `//`, then count the first place too.",
        ],
        why="For `out_size(28, 5, 2, 0)`: the filter can move 23 places to the right of its first place; in steps of 2 that is 11 steps, rounded down, and with the first place 12 places.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_conv2d():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A convolution with stride and padding

        Complete `conv2d(img, K, s, p)`: the convolution of the square image `img` with the square filter `K`, with stride `s` and padding `p`. Replace the three `...`, following steps 1 to 3 of the recipe. The check calls `conv2d([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], [[1, 0], [0, -1]], 2, 0)`, which should give `[[0, -1], [2, -2]]`, and tries other filters, strides and paddings.
        """),
        L.task("Make `conv2d` slide a filter over an image with stride and padding."),
    ])
    return


@app.cell(hide_code=True)
def exercise_conv2d():
    box_conv2d = L.code_box("conv2d", lines=17)
    box_conv2d
    return (box_conv2d,)


@app.cell(hide_code=True)
def check_conv2d(box_conv2d):
    result_conv2d = L.check("conv2d", box_conv2d.value)
    result_conv2d
    return (result_conv2d,)


@app.cell(hide_code=True)
def hints_conv2d():
    L.hints(
        "conv2d",
        [
            "Step 1 is an array of zeros with `np.zeros`, `n + 2 * p` by `n + 2 * p`; the line under it puts the image in the middle. Step 2 is your `out_size` formula.",
            "Step 3 is the line of `convolve`, with the piece taken from `x`, starting at row `r * s` and column `c * s`.",
        ],
        why="With stride 2, output place 1 starts at row 2 of the padded image, so the start of a piece is `r * s`, and it ends `k` further. Taking the piece from `x` instead of `img` brings in the zeros of the frame at the edge.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** `convolve` from [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en) was the case $s = 1$, $p = 0$: its output size, $H - k + 1$, is today's formula with $n = H$ and those values. Kernel size, stride and padding are hyperparameters of a layer, chosen before training; the weights of the filter are what the layer learns. In the next lesson, *Channels and pooling*, a layer gets many filters at once, each making its own output, and pooling shrinks the image while keeping the strongest signals.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_receptive, result_out_size, result_conv2d):
    L.wrap_up(choice_receptive, result_out_size, result_conv2d)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
