# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-channels-pooling. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-channels-pooling", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_odd():
    quick_odd = L.choice(
        "quick-odd",
        question="Why are filters almost always odd in size, such as 3 by 3 or 5 by 5?",
        options={
            "right": "So padding can be equal on both sides and keep the size",
            "fast": "Because odd sizes are faster for a computer to multiply",
            "center": "Because an even filter cannot slide over an image at all",
        },
        correct="right",
        why={
            "fast": "The speed hardly depends on odd or even. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "center": "An even filter slides fine; only the padding to keep the size cannot be the same on both sides. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "right": "With $k = 3$ and $p = 1$, the output keeps the size of the input. Many CNNs keep the size of their layers this way, and let pooling shrink the image on purpose.",
        },
    )
    quick_odd
    return (quick_odd,)


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="What does a stride of 2 do to the width of a layer's output, compared with a stride of 1?",
        options={
            "double": "It doubles it: the filter covers more ground per step",
            "same": "Nothing: the stride only changes how the filter moves",
            "right": "It about halves it: the filter skips every other place",
        },
        correct="right",
        why={
            "double": "The filter takes bigger steps, so it stops at fewer places: fewer numbers come out. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "same": "The stride is in the formula for the output size: $\\lfloor (n + 2p - k)/s \\rfloor + 1$. See [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "right": "Today you meet another way to make an image smaller: pooling, which has no weights at all.",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    One filter finds one kind of pattern, such as a vertical edge. An image holds many kinds: edges in every direction, corners, spots, textures. A real layer has dozens of filters, each with its own feature map, and the next layer combines those maps into larger patterns. And a color photo is not one grid of numbers but three: red, green and blue. Meanwhile, the network has to get from a large image to one answer, so somewhere the grids must get smaller, without losing what was found.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "rgb-filters",
        question="A convolutional layer has 64 filters of 3 by 3 and works on a color photo, with 3 channels: red, green and blue. How many weights do its filters have together, without bias terms?",
        options={
            "flat": "576: 64 filters of 3 × 3",
            "right": "1,728: 64 filters of 3 × 3 × 3",
            "depends": "It depends on the photo: more pixels need more weights",
        },
        reveal="**1,728.** Each filter has a slice of 3 by 3 for every input channel, so 3 × 3 × 3 = 27 weights, and 64 filters have 64 × 27 = 1,728. With a bias term per filter, the layer has 1,792 parameters, the same for a photo of 32 by 32 pixels as for one of 1,000 by 1,000.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An image with $C_\text{in}$ **channels**, such as red, green and blue, has shape $(C_\text{in}, n, n)$ here: the channels come first, so `x[i]` is channel `i`. The photos of [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en) had them last, with shape (rows, columns, 3). A filter for it has a slice of $k$ by $k$ for every channel: shape $(C_\text{in}, k, k)$. At every place, each slice multiplies the piece of its own channel, and all products of all channels are added up into one number. One filter thus makes one feature map, an **output channel**. A layer with $C_\text{out}$ filters, of shape $(C_\text{out}, C_\text{in}, k, k)$ together, makes $C_\text{out}$ output channels, and they are the input channels of the next layer:
        """),
        L.formula(r"\begin{aligned} & [[hidden:\text{out}[o, r, c]]] = \sum_{i=0}^{C_\text{in}-1} \sum_{a=0}^{k-1} \sum_{b=0}^{k-1} \\ & \qquad x[i, r+a, c+b] \cdot [[param:W[o, i, a, b]]] \end{aligned}"),
        mo.md(r"""
        In NumPy, `x[:, r:r + k, c:c + k]` takes the piece at `r`, `c` from every channel at once: shape $(C_\text{in}, k, k)$, the shape of one filter.

        **Pooling** makes every channel smaller. **Max pooling** cuts a channel into blocks of $s$ by $s$, next to each other, and keeps the largest number of each block: the strongest signal survives, even when it moves a pixel within its block. Pooling of 2 by 2 halves the width and the height, and it has no weights: it learns nothing. **Average pooling** keeps the mean of each block instead.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **output channel** | | the feature map one filter makes; the input channel of the next layer | `out[o]` |
        | **pooling** | | making a channel smaller by summarizing each block in one number | |
        | **max pooling** | | pooling that keeps the largest number of each block | `.max()` |
        | **average pooling** | | pooling that keeps the mean of each block | `.mean()` |
        | $C_\text{in}$, $C_\text{out}$ | C in, C out | the number of input channels and of output channels | `len(x)`, `len(W)` |
        """),
        L.watch_out("A filter is not 2D: it always covers all input channels, so it has $C_\\text{in} \\times k \\times k$ weights. And pooling works on every channel on its own; it never mixes channels. Mixing channels is the job of the filters."),
        mo.md(r"""
        The picture: max pooling of 2 by 2 on one channel of 4 by 4. In each of the four blocks, the largest number, in a box with a thick border, is the one that stays; the rest goes gray.
        """),
        C.pool_picture({"title": "Max pooling of 2 by 2", "input": "one channel", "output": "after pooling",
                        "input_caption": "(4, 4)", "output_caption": "(2, 2)"}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A layer with channels, then max pooling:

    1. Check the shapes: input $(C_\text{in}, n, n)$, filters $(C_\text{out}, C_\text{in}, k, k)$, output $(C_\text{out}, m, m)$ with $m = n - k + 1$, here without padding and with stride 1.
    2. Output channel `o` at `r`, `c`: the piece at `r`, `c` of every channel, `x[:, r:r + k, c:c + k]`, times filter `o`, number by number, all added up.
    3. Max pooling with blocks of $s$ by $s$: for every channel, block `r`, `c` starts at row `r * s` and column `c * s`, and its `.max()` stays; for C channels, the output has shape `(C, n // s, n // s)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 2 for an input of 2 channels of 4 by 4 and one filter of shape (2, 3, 3): a diagonal in each slice, from the top left in the first and from the top right in the second. The comments with `#` say which step of the recipe each part is. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_channels():
    import numpy as np

    _x = np.array([[[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [1, 0, 1, 2]],
                   [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]])
    _W = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
    _k = 3
    # 1. the shapes: (2, 4, 4) and (2, 3, 3) give one output channel of 2 by 2
    print(_x.shape, _W.shape)
    _m = len(_x[0]) - _k + 1
    _out = np.zeros((_m, _m))
    # 2. the filter on the piece of every channel, all added up
    for _r in range(_m):
        for _c in range(_m):
            _out[_r, _c] = (_x[0, _r:_r + _k, _c:_c + _k] * _W[0]).sum()  # complete: every channel, _x[:, ...] * _W
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the filter only looks at channel 0, with its first slice: `[[2, 2], [2, 3]]`, the sums along the diagonals from the top left. Completed, with `(_x[:, _r:_r + _k, _c:_c + _k] * _W).sum()`, the second channel adds its own diagonal sums, `[[0, 3], [3, 0]]`: together `[[2, 5], [5, 3]]`. The second channel is a checkerboard: along the anti-diagonal of the filter it holds three 1s at two of the four places, and three 0s at the other two. The one output channel now holds what both input channels show.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The weights of a layer
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights",
        question="A layer has 16 filters of 3 by 3 and gets an input of 8 channels. How many weights do its filters have, without bias terms?",
        options={
            "flat": "144: 16 filters of 3 × 3 each",
            "in": "72: 8 channels of 3 × 3 each",
            "right": "1,152: 16 filters of 8 × 3 × 3",
        },
        correct="right",
        why={
            "flat": "Each filter covers all 8 input channels: a slice of 3 by 3 for each.",
            "in": "That is one filter. The layer has 16, one for every output channel.",
            "right": "$C_\\text{out} \\times C_\\text{in} \\times k \\times k = 16 \\times 8 \\times 9 = 1{,}152$, plus 16 bias terms: the layer's output has 16 channels.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_conv_channels():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A layer with channels

        Complete `conv_channels(x, W)`: the input `x` has shape $(C_\text{in}, n, n)$ and the filters `W` shape $(C_\text{out}, C_\text{in}, k, k)$; it gives the $C_\text{out}$ output channels, without padding and with stride 1. Replace the `...` with step 2 of the recipe. The check calls `conv_channels` with an input of 2 channels of 3 by 3 and 2 filters of shape (2, 2, 2), which should give `[[[4, 6], [4, 5]], [[4, 5], [3, 6]]]`, and tries other inputs, also of 1 and of 3 channels.
        """),
        L.task("Make `conv_channels` give one output channel per filter, from all input channels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_conv_channels():
    box_conv_channels = L.code_box("conv-channels", lines=16)
    box_conv_channels
    return (box_conv_channels,)


@app.cell(hide_code=True)
def check_conv_channels(box_conv_channels):
    result_conv_channels = L.check("conv-channels", box_conv_channels.value)
    result_conv_channels
    return (result_conv_channels,)


@app.cell(hide_code=True)
def hints_conv_channels():
    L.hints(
        "conv-channels",
        [
            "Step 2 of the recipe, like the completed line of the worked example. There, `_W` was one filter; here `W` holds all $C_\\text{out}$ filters, one for each output channel.",
            "The filter of output channel `o` is `W[o]`, with shape $(C_\\text{in}, k, k)$: the same shape as the piece of every channel at `r`, `c`, so the two multiply number by number.",
        ],
        why="Output channel `o` comes from filter `o` alone, and every number of it adds products from all input channels: that is how a layer mixes the channels before it into new ones.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_max_pool():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Max pooling

        Complete `max_pool(x, size)`: the input `x` has shape $(C, n, n)$; it gives the max pooling of every channel, with blocks of `size` by `size` next to each other. Replace the `...` with step 3 of the recipe. The check calls `max_pool([[[2, 0, 1, 3], [1, 4, 0, 2], [3, 1, 5, 0], [0, 2, 1, 1]]], 2)`, which should give `[[[4, 3], [3, 5]]]`, and tries other channels and block sizes.
        """),
        L.task("Make `max_pool` keep the largest number of every block, per channel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_max_pool():
    box_max_pool = L.code_box("max-pool", lines=13)
    box_max_pool
    return (box_max_pool,)


@app.cell(hide_code=True)
def check_max_pool(box_max_pool):
    result_max_pool = L.check("max-pool", box_max_pool.value)
    result_max_pool
    return (result_max_pool,)


@app.cell(hide_code=True)
def hints_max_pool():
    L.hints(
        "max-pool",
        [
            "Block `r`, `c` of a channel starts at row `r * size` and column `c * size`, and is `size` wide.",
            "Take that block from channel `ch` of `x`, as in the convolution of [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en) with a stride of `size`, and keep its `.max()`.",
        ],
        why="Pooling moves in steps of the block size and keeps the largest number instead of a weighted sum: the blocks do not overlap, and every number of the input belongs to exactly one block, or to none at a ragged edge.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The convolution of [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en) now has channels: a filter covers all input channels, and a layer makes one output channel per filter. Pooling moves in steps of the block size and keeps the maximum instead of a weighted sum. A CNN alternates the two: convolutions find patterns, pooling shrinks the image, until a few numbers per channel are left for the last layers. In the next lesson, *Convolution as one big matrix multiplication*, you rewrite all these loops as one matrix multiplication, one of the ways a GPU computes a convolution.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_conv_channels, result_max_pool):
    L.wrap_up(choice_weights, result_conv_channels, result_max_pool)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
