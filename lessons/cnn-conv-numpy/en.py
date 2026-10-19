# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-conv-numpy. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-conv-numpy", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Channels and pooling](/course/open/cnn-channels-pooling?lang=en) and [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_channels():
    quick_channels = L.choice(
        "quick-channels",
        question="A layer gets an input of shape (3, 10, 10) and has 6 filters of 3 by 3, without padding and with stride 1. What shape does its output have?",
        options={
            "per-in": "(3, 8, 8): one output channel per input channel",
            "right": "(6, 8, 8): one output channel per filter",
            "both": "(18, 8, 8): one for every filter and input channel",
        },
        correct="right",
        why={
            "per-in": "Every filter covers all 3 input channels at once and adds them up. See [Channels and pooling](/course/open/cnn-channels-pooling?lang=en).",
            "both": "A filter does not make a channel per input channel: it adds the products of all of them into one number. See [Channels and pooling](/course/open/cnn-channels-pooling?lang=en).",
            "right": "Each of the 6 filters gives one output channel of $10 - 3 + 1 = 8$ by 8. Each of those $6 \\times 64$ numbers is one sum of 27 products: today you compute them all at once.",
        },
    )
    quick_channels
    return (quick_channels,)


@app.cell(hide_code=True)
def quick_check_matmul():
    quick_matmul = L.choice(
        "quick-matmul",
        question="`A` and `B` are matrices whose sizes fit. What is the number in row `i` and column `j` of `A @ B`?",
        options={
            "same-place": "The number at row `i`, column `j` of `A`, times the one at the same place of `B`",
            "sums": "The sum of row `i` of `A`, times the sum of column `j` of `B`",
            "right": "The dot product of row `i` of `A` and column `j` of `B`",
        },
        correct="right",
        why={
            "same-place": "That is `A * B`, number by number. See [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "sums": "The numbers are first multiplied in pairs, and the products are added: a dot product. See [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "right": "And because column `j` of `B.T` is row `j` of `B`, row `i` of `A @ B.T` holds the dot products of row `i` of `A` with every row of `B`.",
        },
    )
    quick_matmul
    return (quick_matmul,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The loops of [Channels and pooling](/course/open/cnn-channels-pooling?lang=en) compute every number of the output on its own. For a layer of 16 filters of 3 by 3 on a color image of 32 by 32, the line inside the loops runs 16 × 30 × 30 = 14,400 times, and Python spends a little time on each. A computer is fastest when it gets one big, regular job, and the job numerical libraries are tuned best for is matrix multiplication: NumPy hands it to a library that has been tuned for decades, and a GPU does thousands of its multiplications at the same time. So people write a convolution as one.
    """)
    return


@app.cell(hide_code=True)
def predict_copies():
    guess_copies = L.guess(
        "copies",
        question="A color photo of 32 by 32 pixels holds 3 × 32 × 32 = 3,072 numbers. For filters of 3 by 3, with padding 1 and stride 1, you write every piece the filter covers as one row of a new matrix. How many numbers does that matrix hold?",
        options={
            "same": "3,072: the same numbers, only in another order and shape",
            "right": "27,648: 9 times as many, since a pixel lands in up to 9 pieces",
            "per-piece": "82,944: 27 times as many, since every piece has 27 numbers",
        },
        reveal="**27,648, exactly 9 times as many.** Padding 1 keeps 32 × 32 = 1,024 places, and the piece at each place has 3 × 3 × 3 = 27 numbers: 1,024 × 27 = 27,648. A pixel away from the edge sits in the pieces of 9 places, so it is copied 9 times; the padding zeros make up for the edges.",
    )
    guess_copies
    return (guess_copies,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Every number of a convolution's output is a dot product: the piece of the input under the filter, its $C_\text{in} \cdot k \cdot k$ numbers, with the weights of one filter. That piece is a **patch**. **im2col** writes every patch as one flat row of a matrix $\mathbf{P}$, place after place, row by row of the image: $m \cdot m$ rows of $C_\text{in} k^2$ numbers. Every filter becomes one row of $\mathbf{W}$, flattened in the same order: $C_\text{out}$ rows of $C_\text{in} k^2$ numbers. All dot products at once are then one matrix multiplication:
        """),
        L.formula(r"[[hidden:\text{out}]] = [[param:\mathbf{W}]] \, \mathbf{P}^\top"),
        mo.md(r"""
        $\mathbf{W}$ has shape $(C_\text{out}, C_\text{in} k^2)$ and $\mathbf{P}^\top$ shape $(C_\text{in} k^2, m^2)$, so the inner sizes fit and out has shape $(C_\text{out}, m^2)$. Row $o$ of out holds filter $o$ at all $m^2$ places, in the order of the rows of $\mathbf{P}$, and `.reshape(c_out, m, m)` turns each row back into an output channel of $m$ by $m$. The name im2col, short for *image to columns*, comes from the first versions, MATLAB's `im2col` and then Caffe, which put every patch in a column; with rows you get the same matrix, transposed. Stride and padding only change which patches you cut: pad first, and step by $s$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **patch** | | the piece of the input under the filter at one place, of every channel: $C_\text{in} \cdot k \cdot k$ numbers | `x[:, r:r + k, c:c + k]` |
        | **im2col** | image to columns | writing every patch as one row of a matrix, so that the convolution becomes one matrix multiplication | `patches(x, k)` |
        | $\mathbf{P}$ | capital P | the patches as rows: one row per place, shape $(m^2, C_\text{in} k^2)$ | `P` |
        | $\mathbf{W}$ | capital W | the filters as rows: shape $(C_\text{out}, C_\text{in} k^2)$ | `W.reshape(c_out, c_in * k * k)` |
        | **`.reshape(n)`** | reshape | with one number: all numbers in one flat row, channel after channel, row after row | `patch.reshape(c_in * k * k)` |
        """),
        L.watch_out("im2col copies: a pixel away from the edge lands in the patches of $k^2$ places. For 64 channels of 224 by 224 and filters of 3 by 3 with padding 1, $\\mathbf{P}$ holds almost 29 million numbers, against 3.2 million in the input. That is why GPU libraries cut the patches tile by tile while they multiply, instead of holding all of $\\mathbf{P}$."),
        mo.md(r"""
        The picture: one channel of 3 by 3 and a filter of 2 by 2, so 2 × 2 = 4 places and 4 patches of 4 numbers. The patch at row 0, column 1 of the image becomes row 1 of $\mathbf{P}$: places are counted row by row.
        """),
        C.patches_picture({"title": "Every patch becomes a row", "input": "one channel", "output": "P, the patches as rows",
                           "input_note": "patch at row 0, column 1", "output_note": "row 1"}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A convolution as one matrix multiplication, without padding and with stride 1:

    1. Count the places per side: $m = n - k + 1$.
    2. For every place, row by row: the patch `x[:, r:r + k, c:c + k]`, flat with `.reshape(c_in * k * k)`, is the next row of $\mathbf{P}$.
    3. Every filter as one row, in the same order: `W.reshape(c_out, c_in * k * k)`.
    4. Multiply, `Wm @ P.T`, and give every filter its output channel: `.reshape(c_out, m, m)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One channel of 4 by 4 and two filters of shape (1, 2, 2): a diagonal, and a small difference between the top right and the bottom left. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_im2col():
    import numpy as np

    _x = np.array([[[1, 2, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 0, 3, 2]]])
    _W = np.array([[[[1, 0], [0, 1]]], [[[0, 1], [-1, 0]]]])
    _c_in, _k = 1, 2
    # 1. the places per side
    _m = len(_x[0]) - _k + 1
    # 2. every patch as one row, place after place
    _rows = []
    for _r in range(_m):
        for _c in range(_m):
            _rows.append(_x[:, _r:_r + _k, _c:_c + _k].reshape(_c_in * _k * _k))
    _P = np.array(_rows)
    print(_P.shape)
    # 3. every filter as one row
    _Wm = _W.reshape(2, _c_in * _k * _k)
    # 4. all dot products at once
    _out = _Wm @ _P.T  # complete: give every filter its output channel, .reshape(2, _m, _m)
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\mathbf{P}$ has shape (9, 4): 3 × 3 places, and 4 numbers in every patch. As it is, the cell prints the product, of shape (2, 9): the first row is the diagonal filter at the 9 places, `[2 4 0 5 2 5 0 5 3]`, and the second row the other filter. Completed, with `(_Wm @ _P.T).reshape(2, _m, _m)`, it prints two output channels of 3 by 3, `[[2 4 0] [5 2 5] [0 5 3]]` and `[[-1 -1 -1] [1 0 -1] [1 1 0]]`: exactly what your `conv_channels` of [Channels and pooling](/course/open/cnn-channels-pooling?lang=en) gives for this input, number for number.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The shapes
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape",
        question="An input of shape (3, 8, 8) and 16 filters of shape (3, 3, 3), without padding and with stride 1. What shape does $\\mathbf{W}\\mathbf{P}^\\top$ have, before the reshape?",
        options={
            "right": "(16, 36): a row for every filter",
            "per-place": "(36, 16): a row for every place",
            "per-pixel": "(16, 64): a column for every pixel",
        },
        correct="right",
        why={
            "per-place": "That is $\\mathbf{P}\\mathbf{W}^\\top$, the same numbers transposed. $\\mathbf{W}$ comes first, so its rows, the filters, are the rows of the result.",
            "per-pixel": "A filter of 3 by 3 fits at $8 - 3 + 1 = 6$ places per side, not at all 8: the result has a column for every place.",
            "right": "$\\mathbf{W}$ is (16, 27) and $\\mathbf{P}^\\top$ is (27, 36): 16 filters, each at 6 × 6 = 36 places. `.reshape(16, 6, 6)` gives the 16 output channels.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_patches():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. im2col

        Complete `patches(x, k)`: the input `x` has shape $(C_\text{in}, n, n)$, and it gives $\mathbf{P}$, every patch for a filter of `k` by `k` as one row, without padding and with stride 1. Replace the `...` following step 2 of the recipe. The check calls `patches` with an input of 2 channels of 3 by 3 and `k` = 2, which should give 4 rows of 8 numbers, `[[1, 2, 0, 1, 0, 1, 1, 0], …]`, and tries other inputs and kernel sizes.
        """),
        L.task("Make `patches` write every patch of the input as one flat row."),
    ])
    return


@app.cell(hide_code=True)
def exercise_patches():
    box_patches = L.code_box("patches", lines=13)
    box_patches
    return (box_patches,)


@app.cell(hide_code=True)
def check_patches(box_patches):
    result_patches = L.check("patches", box_patches.value)
    result_patches
    return (result_patches,)


@app.cell(hide_code=True)
def hints_patches():
    L.hints(
        "patches",
        [
            "Step 2 of the worked example, with `x`, `r`, `c`, `k` and `c_in` instead of the names with `_`.",
            "`x[:, r:r + k, c:c + k]` takes the patch from every channel at once: shape $(C_\\text{in}, k, k)$. One flat row has all $C_\\text{in} \\cdot k \\cdot k$ of its numbers.",
        ],
        why="Every row of $\\mathbf{P}$ is everything one filter looks at in one place, so a single dot product with a filter row gives one output number. The loops only cut; the multiplying comes later, all at once.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_conv_matmul():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The convolution as one `@`

        Complete `conv_matmul(P, W, m)`: `P` is the result of `patches`, `W` holds the filters, of shape $(C_\text{out}, C_\text{in}, k, k)$, and `m` is the number of places per side. It gives the $C_\text{out}$ output channels of $m$ by $m$. Replace the two `...` following steps 3 and 4 of the recipe. The check calls `conv_matmul` with the 4 patches of 2 channels of 3 by 3 and 2 filters of shape (2, 2, 2), which should give `[[[2, 0], [5, 2]], [[2, 2], [4, 1]]]`, and tries other filters and channels.
        """),
        L.task("Make `conv_matmul` compute all output channels with one matrix multiplication."),
    ])
    return


@app.cell(hide_code=True)
def exercise_conv_matmul():
    box_conv_matmul = L.code_box("conv-matmul", lines=11)
    box_conv_matmul
    return (box_conv_matmul,)


@app.cell(hide_code=True)
def check_conv_matmul(box_conv_matmul):
    result_conv_matmul = L.check("conv-matmul", box_conv_matmul.value)
    result_conv_matmul
    return (result_conv_matmul,)


@app.cell(hide_code=True)
def hints_conv_matmul():
    L.hints(
        "conv-matmul",
        [
            "A row of `Wm` must be as long as a row of `P`: `len(P[0])` numbers. Steps 3 and 4 of the worked example show the rest.",
            "`Wm @ P.T` has a row per filter and a column per place, place after place, row by row; `.reshape(c_out, m, m)` cuts each row into `m` rows of `m`.",
        ],
        why="Flattening a filter and flattening a patch use the same order, channel after channel and row after row, so number $j$ of a filter row meets number $j$ of a patch row: the dot product is exactly the sum of [Channels and pooling](/course/open/cnn-channels-pooling?lang=en).",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loops of [Channels and pooling](/course/open/cnn-channels-pooling?lang=en) and the matrix multiplication of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) are the same convolution: every output number is the dot product of a patch and a filter. Even with the patches cut by a Python loop, the matrix version of a layer with 16 filters on a color image of 32 by 32 is more than 10 times faster in NumPy than those loops, and the gap grows with the number of filters. Kumar Chellapilla, Sidd Puri and Patrice Simard wrote CNNs this way in 2006, to run them fast, also on a GPU; cuDNN, the library for deep learning on NVIDIA GPUs, still computes many convolutions as matrix multiplications, cutting the patches tile by tile as it goes. In the next lesson, *Project: your first CNN*, PyTorch's `nn.Conv2d` does all of this for you, and you train a small CNN on Fashion-MNIST.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_patches, result_conv_matmul):
    L.wrap_up(choice_shape, result_patches, result_conv_matmul)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
