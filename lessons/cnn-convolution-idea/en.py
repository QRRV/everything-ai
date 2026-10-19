# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-convolution-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-convolution-idea", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en) and [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_piece():
    quick_piece = L.choice(
        "quick-piece",
        question="`img` is an image of shape (28, 28). What is the shape of `img[5:8, 10:13]`?",
        options={
            "right": "(3, 3)",
            "stop": "(4, 4): from 5 up to and including 8",
            "values": "(8, 13)",
        },
        correct="right",
        why={
            "stop": "A slice stops just before its end: `5:8` is rows 5, 6 and 7. See [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en).",
            "values": "The numbers in a slice say where it starts and stops, not how big it is. See [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en).",
            "right": "Rows 5, 6 and 7, and columns 10, 11 and 12: a piece of 3 by 3 pixels.",
        },
    )
    quick_piece
    return (quick_piece,)


@app.cell(hide_code=True)
def quick_check_neuron():
    quick_neuron = L.choice(
        "quick-neuron",
        question="A hidden layer gets an image of 28 by 28 pixels as 784 inputs. How many weights does each of its neurons have, without its bias term?",
        options={
            "one": "1: one weight per neuron",
            "nine": "9: one per pixel and the pixels around it",
            "right": "784: one per pixel",
        },
        correct="right",
        why={
            "one": "A neuron makes a weighted sum of all its inputs, each with its own weight. See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "nine": "A neuron of a hidden layer sees all its inputs, not a small patch of them. See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "right": "Each neuron makes a weighted sum of all 784 inputs, so it has a row of 784 weights in `W`.",
        },
    )
    quick_neuron
    return (quick_neuron,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A neuron of a hidden layer has one weight for every pixel. A stroke at the top left of an image and the same stroke at the bottom right meet completely different weights, so the network has to learn that stroke again for every place it can be. Yet an edge, a corner or a loop looks the same wherever it is. Better to look for it with one small set of weights, the same at every place in the image.

    **An everyday example.** To find every "the" in a book, you do not learn a separate rule for page 1, page 2 and so on. You slide the same three letters along every line, and note where they match.
    """)
    return


@app.cell(hide_code=True)
def predict_places():
    guess_places = L.guess(
        "places",
        question="A filter of 3 by 3 pixels slides over an image of 28 by 28 pixels, to every place where it fits completely. How many places do you think that is?",
        options={
            "all": "784: one for every pixel",
            "right": "676",
            "nine": "9: as many as the filter has pixels",
        },
        reveal="**676: 26 by 26.** Along a row, the filter's left edge can start at columns 0 to 25: at 26 it would stick out past column 27. The same holds for the rows. So the result is a grid of 26 by 26: two rows and two columns smaller than the image.",
    )
    guess_places
    return (guess_places,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **filter** is a small grid of weights, `K`, of $k$ by $k$, often 3 by 3. A **convolution** slides it over the image. At every place where it fits, it multiplies each pixel under it with the weight on top of it, and adds the products up. That sum is one number of the **feature map**, `out`:
        """),
        L.formula(r"[[hidden:\text{out}[r, c]]] = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} \text{img}[r+i, c+j] \cdot [[param:K[i, j]]]"),
        mo.md(r"""
        In NumPy, that is `(img[r:r + k, c:c + k] * K).sum()`: the piece of the image of $k$ by $k$ that starts at row $r$ and column $c$, times the filter, number by number, then the sum. An image of $H$ rows by $W$ columns, its height and its width, gives a feature map of $H - k + 1$ by $W - k + 1$.

        Two ideas make this different from a hidden layer:

        - **Weight sharing.** The same $k \cdot k$ weights are used at every place: a filter of 3 by 3 has 9 weights for the whole image, where a neuron of a hidden layer has 784.
        - **Locality.** Each number of the feature map depends only on a small piece of the image, the $k$ by $k$ pixels under the filter, not on all of them.

        A feature map is large where the image looks like the filter. The filter `[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]` takes the column on the right minus the column on the left. So it gives a large positive number where the numbers go from low on the left to high on the right: a vertical edge. The other way round, from high to low, it gives a large negative number; on an even surface, 0. In a network that learns, the weights of the filters are learned too. Here you set them by hand. Strictly, in mathematics a convolution first turns the filter upside down and back to front; deep learning skips that step and still calls it a convolution.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **filter** | | a small grid of weights, $k$ by $k$, that slides over an image; also called a kernel | `K` |
        | $k$ | k | the size of the filter: $k$ rows and $k$ columns | `k = len(K)` |
        | **convolution** | | sliding a filter over an image and, at every place, adding up the products of pixels and weights | `(img[r:r + k, c:c + k] * K).sum()` |
        | **feature map** | | the grid of sums a convolution gives: large where the image looks like the filter | `out` |
        | **weight sharing** | | the same weights at every place of the input | |
        | **locality** | | each output depends only on a small patch of the input | |
        | **`np.zeros((rows, cols))`** | | a grid of zeros with `rows` rows and `cols` columns; the shape goes in as a tuple, hence the two pairs of parentheses | `np.zeros((2, 3))` |
        | **`out[r, c] = x`** | | puts the number `x` in row `r` and column `c` of `out`; with a slice, in every place of that part: `img[:, 3:] = 1` | `out[0, 1] = 5` |
        """),
        L.watch_out("\"Kernel\" means several things in AI. For a convolution, it is another word for the filter. But the Python process behind a marimo notebook is also called a kernel, and so is a small program that runs on a GPU (graphics processing unit). This course says filter for the weights of a convolution."),
        mo.md(r"""
        The picture: image 0 of QMNIST, a set of handwritten digits that the course keeps in `data/qmnist-small.npz`. It is a 5. Next to it are its feature maps for the vertical edge filter above and for `[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]`, which looks for a horizontal edge with low numbers above and high numbers below. Magenta is a large positive number, gray a large negative one.
        """),
        mo.Html(C.picture({"title": "One digit, two filters", "image": "the image", "vertical": "vertical edges",
                           "horizontal": "horizontal edges",
                           "alt": "A handwritten 5, and two feature maps of 26 by 26. The map of the vertical edge filter is magenta along the left side of each stroke and gray along its right side; the map of the horizontal edge filter is magenta along the top of each stroke and gray along its bottom."})),
        mo.md(r"""
        In the numbers, a pixel is 1 for ink and 0 for paper. So the map of the vertical edge filter is magenta on the left side of every stroke, where 0 for paper turns to 1 for ink, and gray on its right side. The map of the horizontal filter does the same above and below each stroke. The flat top of the 5 shows up clearly in the horizontal map; its upright stroke in the vertical one.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To convolve an image with a filter:

    1. Choose a filter `K` of $k$ by $k$.
    2. Make the feature map, all zeros, of $H - k + 1$ by $W - k + 1$: `out = np.zeros((rows, cols))`.
    3. For every row `r` and column `c` of `out`: `out[r, c] = (img[r:r + k, c:c + k] * K).sum()`.
    4. Read the feature map: large positive numbers where the image looks like the filter.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An image of 6 by 6 with 0 on the left and 1 on the right, with the vertical edge filter; then the same steps on the 5 from the picture, which `C.digit(0)` gives as an array of 28 by 28. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_convolve():
    import numpy as np
    import matplotlib.pyplot as plt

    _img = np.zeros((6, 6))
    _img[:, 3:] = 1   # 0 on the left, 1 on the right
    # 1. a filter of 3 by 3 for a vertical edge: low on the left, high on the right
    _K = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    _k = len(_K)
    # 2. the feature map, all zeros: 6 - 3 + 1 = 4 by 4
    _out = np.zeros((len(_img) - _k + 1, len(_img[0]) - _k + 1))
    # 3. at every place: the piece times the filter, added up
    for _r in range(len(_out)):
        for _c in range(len(_out[0])):
            _out[_r, _c] = (_img[_r:_r + _k, _c:_c + _k] * _K).sum()
    print(_out)

    # the same steps on a handwritten digit
    _digit = C.digit(0)
    _out = np.zeros((len(_digit) - _k + 1, len(_digit[0]) - _k + 1))
    for _r in range(len(_out)):
        for _c in range(len(_out[0])):
            _out[_r, _c] = (_digit[_r:_r + _k, _c:_c + _k] * _K).sum()
    # 4. read the feature map: where is it large?
    print(_out.shape, _out.max(), _out.min())
    plt.imshow(_out, cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the 6 by 6 image, the feature map is 0 in its first and last column, where the filter sees only 0s or only 1s, and 3 in the two middle columns, where the edge runs under it: three 1s on the right times +1, three 0s on the left times −1. On the digit, the feature map is 26 by 26. Its largest numbers lie along the left side of the strokes and its most negative along their right side; with `cmap="gray"`, those are the white and the black lines in the image.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many weights?
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights",
        question="A filter of 5 by 5 slides over an image of 28 by 28 and makes a feature map of 24 by 24. How many weights does it use for the whole feature map?",
        options={
            "per-place": "14,400: 25 weights at each of the 576 places",
            "right": "25",
            "per-pixel": "784: one per pixel",
        },
        correct="right",
        why={
            "per-place": "The filter brings the same 25 weights to every place. See *Weight sharing* in *The idea*.",
            "per-pixel": "That is what one neuron of a hidden layer has. See *Weight sharing* in *The idea*.",
            "right": "5 · 5 = 25 weights, the same at all 576 places: weight sharing. A neuron of a hidden layer has 784 weights for one number."
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_convolve():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A convolution for any filter

        `convolve(img, K)` convolves an image with a filter of any size $k$ by $k$, and gives back the feature map. Replace the three `...`: how many places the filter fits down, `rows`, and across, `cols`, and for each place the sum of the image piece times the filter. The check calls `convolve([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], [[1, 0], [0, -1]])`, a filter of 2 by 2, which should give `[[0, -1, -1], [-1, 1, 3], [2, 0, -2]]`. It also tries filters of 1 by 1 and 3 by 3, and images that are not square.
        """),
        L.task("Make `convolve` slide a filter of any size over the image, and give back the feature map."),
    ])
    return


@app.cell(hide_code=True)
def exercise_convolve():
    box_convolve = L.code_box("convolve", lines=14)
    box_convolve
    return (box_convolve,)


@app.cell(hide_code=True)
def check_convolve(box_convolve):
    result_convolve = L.check("convolve", box_convolve.value)
    result_convolve
    return (result_convolve,)


@app.cell(hide_code=True)
def hints_convolve():
    L.hints(
        "convolve",
        [
            "Steps 2 and 3 of the recipe, with `k` instead of 3.",
            "A filter of `k` fits at as many places down as the image has rows, minus `k`, plus 1; across, the same with the length of a row. At each place, take the piece of `k` rows and `k` columns that starts there.",
            "`rows = len(img) - k + ___`, `cols = len(img[0]) - ___ + 1`, and `out[r, c] = (img[r:r + k, c:c + ___] * K).sum()`.",
        ],
        why="With a filter of 2 by 2 on an image of 4 by 4, the filter fits at 3 by 3 places. At the top left, the piece is `[[1, 2], [0, 1]]`, and times the filter `[[1, 0], [0, -1]]` that gives 1 · 1 + 1 · (−1) = 0. The other places go the same way.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_filter():
    mo.md(r"""
    ### Exercise 3. Which filter?
    """)
    return


@app.cell(hide_code=True)
def choice_filter_cell():
    choice_filter = L.choice(
        "which-filter",
        question="You look for horizontal edges with paper above and ink below: a pixel of 0 above, 1 below. Which filter gives a large positive number there?",
        options={
            "vertical": "`[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]`",
            "flipped": "`[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]`",
            "right": "`[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]`",
            "flat": "`[[1, 1, 1], [1, 1, 1], [1, 1, 1]]`",
        },
        correct="right",
        why={
            "vertical": "This filter compares the left with the right, not the top with the bottom. See the edge filter in *The idea*.",
            "flipped": "Here the ink below meets the −1s, so this filter gives a large negative number there. It is large where ink is above and paper below. See the edge filter in *The idea*.",
            "flat": "This filter adds up all nine pixels: it is large wherever there is much ink, edge or not. See *The idea*.",
            "right": "The top row weighs the pixels above with −1, the bottom row the pixels below with +1: paper above and ink below give three 0s times −1 plus three 1s times +1, which is 3, and an even surface gives 0.",
        },
    )
    choice_filter
    return (choice_filter,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A convolution works on images as the grids of numbers of [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en), with the slices of [Selecting parts of an array](/course/open/np-indexing?lang=en). Where a hidden layer of [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en) gives every neuron one weight per pixel, a filter shares a few weights over the whole image, and looks at one small piece at a time. A convolutional neural network, a CNN, stacks many filters and learns their weights. In the next lesson, *Recurrence: a network with memory*, the same idea of reusing weights works along a sentence instead of across an image.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_filter, choice_weights, result_convolve):
    L.wrap_up(choice_weights, result_convolve, choice_filter)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
