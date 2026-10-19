# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-matrix-calculus. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-matrix-calculus", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Partial derivatives and the gradient, formally](/course/open/math-partial-derivatives?lang=en) and [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_paths():
    quick_paths = L.choice(
        "quick-paths",
        question="$z = u + v$, with $u = 2t$ and $v = t^2$. What is $\\frac{dz}{dt}$ at $t = 3$?",
        options={
            "right": "8: the two paths added, 2 + 6",
            "times": "12: the two paths multiplied, 2 · 6",
            "one": "6: the path through $v$, the larger one",
        },
        correct="right",
        why={
            "times": "Along one path you multiply, but the paths themselves are added. See [Partial derivatives and the gradient, formally](/course/open/math-partial-derivatives?lang=en).",
            "one": "$t$ reaches $z$ through $u$ too, and every path counts. See [Partial derivatives and the gradient, formally](/course/open/math-partial-derivatives?lang=en).",
            "right": "The path through $u$ gives $1 \\cdot 2$, the path through $v$ gives $1 \\cdot 2t = 6$: together 8.",
        },
    )
    quick_paths
    return (quick_paths,)


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`A = np.array([[1, 2, 3], [4, 5, 6]])`. What is `A.T[2, 0]`?",
        options={
            "six": "6",
            "right": "3",
            "error": "An error: `A` has only 2 rows",
        },
        correct="right",
        why={
            "six": "6 is in column 2 of `A` too, but in row 1. Row 2 of `A.T` is column 2 of `A`, `[3, 6]`, and its first number is 3. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "error": "`A.T` has the shape (3, 2), so it does have a row 2: column 2 of `A`. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "right": "Row 2 of `A.T` is column 2 of `A`, `[3, 6]`: `A.T[2, 0]` is `A[0, 2]`, which is 3.",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A layer of a network computes $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ for a whole batch at once, as in [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en), here without the bias terms. With 3 features and 2 outputs, $\mathbf{W}$ already has 6 numbers, and a real layer has thousands. To train it, you need the partial derivative of the loss for every one of them. One by one, that is endless bookkeeping. With matrices, a single product gives them all, and the shapes tell you which product.

    **An everyday example.** When you put together flat-pack furniture, the shape of each part tells you where it goes: a board with holes along one edge only fits one way round. You hardly need the manual.
    """)
    return


@app.cell(hide_code=True)
def predict_shape():
    guess_shape = L.guess(
        "shape-of-grad",
        question="$\\mathbf{W}$ has the shape (2, 3): one row per output, one column per feature. What shape do you think the gradient of the loss for $\\mathbf{W}$ has?",
        options={
            "flipped": "(3, 2): the transpose of $\\mathbf{W}$",
            "one": "(1,): one number, like the loss",
            "right": "(2, 3): one number per weight",
        },
        reveal="**(2, 3).** The gradient holds one partial derivative for every number of $\\mathbf{W}$, and puts it in the same place: the number in row $k$, column $j$ says how the loss changes when you nudge $W_{kj}$. So it has the shape of $\\mathbf{W}$. The loss is one number, but it has 6 partial derivatives here. Some books write this gradient transposed; the course, like PyTorch, keeps the shape of $\\mathbf{W}$.",
    )
    guess_shape
    return (guess_shape,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Matrix calculus** works out derivatives for whole vectors and matrices at once. The gradient of the loss $\mathcal{L}$ for a matrix $\mathbf{W}$, written $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$, holds one partial derivative for every number of $\mathbf{W}$, in the same place: it has the shape of $\mathbf{W}$.

        Take a layer $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$, with $\mathbf{X}$ of shape $(n, d)$: $n$ examples with $d$ features; $\mathbf{W}$ of shape $(m, d)$: one row per output; and $\mathbf{Y}$ of shape $(n, m)$. Say the gradient for $\mathbf{Y}$ is known: $\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$, with the shape of $\mathbf{Y}$. A weight $W_{kj}$ reaches the loss through column $k$ of $\mathbf{Y}$, once for every example $i$: $Y_{ik}$ contains $X_{ij} \cdot W_{kj}$, so the local derivative is $X_{ij}$. The multivariable chain rule adds the paths: $\frac{\partial \mathcal{L}}{\partial W_{kj}} = \sum_i G_{ik} \, X_{ij}$, and that is the number in row $k$, column $j$ of $\mathbf{G}^\top \mathbf{X}$. In the same way for $\mathbf{X}$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{W}}]] = [[grad:\mathbf{G}]]^\top [[data:\mathbf{X}]] \qquad [[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{X}}]] = [[grad:\mathbf{G}]] \, [[param:\mathbf{W}]]"),
        mo.md(r"""
        You do not have to write out that sum every time: the shapes give it away. $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ must have the shape $(m, d)$. From $\mathbf{G}$, $(n, m)$, and $\mathbf{X}$, $(n, d)$, only one product gives $(m, d)$: $\mathbf{G}^\top \mathbf{X}$, which is $(m, n)$ @ $(n, d)$. The same works for a linear model with the mean squared error, $\mathcal{L} = \frac{1}{n} \sum_i r_i^2$ with the residuals $\mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{w}$: the gradient for the predictions is $-\frac{2}{n} \mathbf{r}$, so the gradient for $\mathbf{w}$ is $-\frac{2}{n} \mathbf{X}^\top \mathbf{r}$, one number per weight. That is the `-2 * (r @ X) / len(y)` you wrote in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en), now found from the shapes.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **matrix calculus** | | derivatives for whole vectors and matrices at once; the gradient for a matrix has the shape of that matrix | |
        | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ | partial L by partial W | the gradient of the loss for the matrix $\mathbf{W}$: one partial derivative per number of $\mathbf{W}$, in its place | `dW` |
        | $\mathbf{G}$ | G | here: the gradient of the loss for the output $\mathbf{Y}$, with the shape of $\mathbf{Y}$ | `G` |
        """),
        L.watch_out("`G.T @ X` and `X.T @ G` both run, but only the first has the shape of `W`; the second is its transpose, the formula for a layer written as `X @ W` with one column per output. When the shapes do not fit, or the result does not have the shape of the matrix it is the gradient for, you have the wrong product. With a vector, such as `r` of shape (4,), `X.T @ r` and `r @ X` give the same."),
        mo.md(r"""
        The picture: the shapes of $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ for 4 examples, 3 features and 2 outputs, and the two products that give the gradients. The size of each box follows its shape.
        """),
        mo.Html(C.shapes_picture({"title": "Follow the shapes", "forward": "the layer",
                                  "for_w": "the gradient for W, with the shape of W",
                                  "for_x": "the gradient for X, with the shape of X",
                                  "alt": "Three rows of boxes, each drawn to its shape. First row: X (4, 3) times W transposed (3, 2) gives Y (4, 2). Second row: G transposed (2, 4) times X (4, 3) gives dW (2, 3), the shape of W. Third row: G (4, 2) times W (2, 3) gives dX (4, 3), the shape of X."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the gradient of the loss for a matrix:

    1. Write down the shape of every matrix: the input, the weights, the output, and the gradient for the output.
    2. The gradient for a matrix has the shape of that matrix.
    3. Find the product of the other two that gives exactly that shape: transpose one where needed, so that the inner numbers match.
    4. Check one number with a tiny step: nudge one entry of the matrix, and compare.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A layer with 4 examples, 3 features and 2 outputs, and as the loss half the sum of the squared differences with the labels `_T`: for that loss, the gradient for $\mathbf{Y}$ is exactly $\mathbf{Y} - \mathbf{T}$. The comments with `#` say which step of the recipe each part is. Step 4 checks one number; add a check of a second one, in row 0, column 2, where the comment says, and run the cell again.
    """)
    return


@app.cell
def example_shapes():
    import numpy as np

    _X = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])   # 4 examples, 3 features
    _W = np.array([[0.5, 1.0, 0.0], [-1.0, 0.0, 2.0]])   # 2 outputs, one row each, 3 features
    _T = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])   # the labels

    def _loss(W):
        return ((_X @ W.T - _T) ** 2).sum() / 2

    # 1. the shapes: X (4, 3), W (2, 3), and G = Y - T with the shape of Y, (4, 2)
    _G = _X @ _W.T - _T
    print(_X.shape, _W.shape, _G.shape)
    # 2. and 3. dW has the shape of W, (2, 3): (2, 4) @ (4, 3) is G.T @ X
    _dW = _G.T @ _X
    print(_dW)
    # 4. check the number in row 1, column 0: nudge only that weight
    _h = 0.001
    _nudged = _W + np.array([[0.0, 0.0, 0.0], [_h, 0.0, 0.0]])
    print((_loss(_nudged) - _loss(_W)) / _h, _dW[1, 0])
    # 4. check row 0, column 2 the same way: put _h in that place, and print both numbers
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\mathbf{G}^\top \mathbf{X}$ has the shape (2, 3), the shape of $\mathbf{W}$, and the number in row 1, column 0 is −2. The tiny step gives about −1.997: close, so the product is right. In row 0, column 2 the product gives 0.5, and the tiny step about 0.5015. All six partial derivatives came from one product, without a single extra run of the loss.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_x():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The gradient for X
    """)
    return


@app.cell(hide_code=True)
def choice_x_cell():
    choice_x = L.choice(
        "grad-x",
        question="$\\mathbf{Y} = \\mathbf{X}\\mathbf{W}^\\top$, with `X` of shape (5, 3) and `W` of shape (4, 3). `G` is the gradient for $\\mathbf{Y}$. Which gives the gradient for `X`?",
        options={
            "right": "`G @ W`",
            "wg": "`W @ G`",
            "gx": "`G.T @ X`",
        },
        correct="right",
        why={
            "wg": "`W` is (4, 3) and `G` is (5, 4): the inner numbers, 3 and 5, do not match, so this stops with an error.",
            "gx": "That is the gradient for `W`, with the shape (4, 3). The gradient for `X` has the shape of `X`, (5, 3).",
            "right": "`G` is (5, 4) and `W` is (4, 3): together (5, 3), the shape of `X`.",
        },
    )
    choice_x
    return (choice_x,)


@app.cell(hide_code=True)
def text_exercise_w():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Two models at once

        Two linear models predict from the same features, as in [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en): each model is a row of `W`, and `X @ W.T` gives the predictions of both, one column per model. The loss is the mean squared error of the first model plus that of the second. `grad_W(X, W, Y)` should give the gradient of that loss for `W`. Write the body of the function: the residuals `R = Y - X @ W.T`, the number of examples, and the gradient. The check calls `grad_W(np.array([[1, 1, 0], [0, 2, 1], [3, 0, 1], [1, 1, 2]]), np.array([[1, 0, 1], [0.5, 1, 0]]), np.array([[1, 2], [2, 2], [4, 1], [2, 2]]))`, which should give `np.array([[0.5, -0.5, 0.5], [0.25, -0.5, -0.25]])`, and tries other data.
        """),
        L.task("Make `grad_W` give the gradient of the loss for `W`, with the shape of `W`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_w():
    box_w = L.code_box("grad-two", lines=8)
    box_w
    return (box_w,)


@app.cell(hide_code=True)
def check_w(box_w):
    result_w = L.check("grad-two", box_w.value)
    result_w
    return (result_w,)


@app.cell(hide_code=True)
def hints_w():
    L.hints(
        "grad-two",
        [
            "Steps 1 to 3 of the recipe. The gradient for `W` has the shape of `W`: one row per model, one column per feature. `X` is (n, features), and `R` has one row per example and one column per model.",
            "For each model, the gradient for its predictions is $-\\frac{2}{n}$ times its residuals, so for all predictions it is $-\\frac{2}{n} \\mathbf{R}$. Which product of `R` and `X` gives one row per model?",
        ],
        why="Here `R` is `[[0, 0.5], [1, 0], [0, -0.5], [-1, 0.5]]`, and `R.T @ X` gives `[[-1, 1, -1], [-0.5, 1, 0.5]]`. Times $-\\frac{2}{4}$: `[[0.5, -0.5, 0.5], [0.25, -0.5, -0.25]]`. Each row is the gradient of one model, as in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Both gradients of a layer

        `layer_grads(X, W, G)` should give the gradients of the loss for `W` and for `X`, for the layer `Y = X @ W.T`, from the gradient `G` for `Y`. Write the body of the function, following the recipe, and give back `[dW, dX]`. The check calls it with `X` of shape (4, 3), `W` of shape (2, 3) and `G` of shape (4, 2), and tries other shapes too.
        """),
        L.task("Make `dW` the gradient for `W` and `dX` the gradient for `X`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("layer-grads", lines=8)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("layer-grads", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "layer-grads",
        [
            "Step 2 of the recipe: `dW` has the shape of `W`, (2, 3), and `dX` the shape of `X`, (4, 3).",
            "Each gradient is a product of `G` and the other matrix. Write down their shapes: does one of them need a transpose, and does `G` come first or second?",
        ],
        why="`G.T @ X` is (2, 4) @ (4, 3), the shape of `W`; `G @ W` is (4, 2) @ (2, 3), the shape of `X`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The multivariable chain rule of [Partial derivatives and the gradient, formally](/course/open/math-partial-derivatives?lang=en) adds the paths; for a matrix product, those sums are a product with a transpose, as in [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en). The gradient for a matrix has the shape of that matrix, and the shapes tell you which product gives it. In the next lesson, *Rank: how many directions a matrix has*, you see how many directions a matrix really has, however big it is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_x, result_w, result_layer):
    L.wrap_up(choice_x, result_w, result_layer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
