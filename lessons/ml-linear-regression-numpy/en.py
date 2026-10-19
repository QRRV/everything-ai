# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-linear-regression-numpy. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-linear-regression-numpy", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) and [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`X` has shape (8, 3) and `w` has shape (3,). What is the shape of `X @ w + b`?",
        options={
            "inner": "(3,)",
            "right": "(8,)",
            "same": "(8, 3)",
        },
        correct="right",
        why={
            "inner": "That is the inner number. Which number remains? See the shape rule in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "same": "Each row of `X` gives one dot product, and `+ b` adds to each. See the worked example in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "right": "One prediction per row: 8 numbers. `+ b` adds the bias term to each of them.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="The loss of one example is $(y - w \\cdot x)^2$, with the residual $r = y - w \\cdot x$. What is its slope with respect to $w$?",
        options={
            "outer": "$2r$",
            "inner": "$-x$",
            "right": "$2r \\cdot (-x)$",
        },
        correct="right",
        why={
            "outer": "That is the outer slope only. What does the chain rule do with the inner one? See *The idea* in [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "inner": "That is the inner slope only. What is the slope of the outer step, $r^2$? See *The idea* in [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "right": "The outer step $r^2$ has the slope $2r$, the inner step the slope $-x$, and the chain rule multiplies them.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en) you measured each slope by nudging a parameter: one extra loss for every parameter, at every step. With one feature that is fine. A house has many features, and a model for text has millions of weights. With the chain rule you get every slope exactly, in one go, with one matrix multiplication. That is linear regression on many features, and the first piece of your own neural network: it learns in the same way, only with more steps in the chain.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           0, here_label="← today", title="Where your project stands: your own neural network"),
        mo.md(r"""
        **An everyday example.** A sound engineer at a mixing desk with many sliders. Trying each slider on its own to hear what it does takes a long time. An engineer who knows how each slider works moves them all at once, each in the direction that helps.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_nudges():
    guess_nudges = L.guess(
        "nudges",
        question="A model has 1000 weights and a bias term. To measure every slope by nudging, how many times do you think you work out the loss in one step of gradient descent?",
        options={
            "two": "2",
            "thousand": "About 1000",
            "million": "About 1,000,000",
        },
        reveal="**About 1000:** once where you are, and once more for each of the 1001 parameters. Each of those is a pass over all the data. The chain rule gives all 1001 slopes from one pass: no nudging at all.",
    )
    guess_nudges
    return (guess_nudges,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Linear regression predicts with a weighted sum of the features plus a bias term, $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$, and finds $\mathbf{w}$ and $b$ by making the mean squared error as small as possible. You know every piece. For one example, the chain rule gives the slope $2r \cdot (-x)$ for a weight. The MSE is the mean of the losses of all examples. When $w$ moves a tiny step, each loss changes by its own slope times that step, so their mean changes by the mean of those changes: the slope of the MSE is the mean of the slopes. The worked example checks this by measuring. For weight $j$ and for $b$, the slopes are:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] = -\frac{2}{n} \sum_{i=1}^{n} [[residual:r_i]] \, x_{ij} \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b}]] = -\frac{2}{n} \sum_{i=1}^{n} [[residual:r_i]]"),
        mo.md(r"""
        Here $x_{ij}$ is feature $j$ of example $i$. The sum over $i$ multiplies each residual by the number in the same place of column $j$ of $\mathbf{X}$, and adds up: a dot product. For all weights at once, that is `r @ X`. With the vector first, it acts as one row: (n,) times (n, d) gives (d,), one slope per weight.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **linear regression** | | a model that predicts with a weighted sum of the features plus a bias term, fitted by making the mean squared error as small as possible | `X @ w + b` |
        | $x_{ij}$ | x i j | feature $j$ of example $i$: row $i$, column $j$ of $\mathbf{X}$ | `X[i, j]` |
        | `np.zeros(d)` | | an array of $d$ zeros, such as the weights before training | `np.zeros(3)` |
        """),
        L.watch_out("`r @ X`, not `X @ r`. `X` has shape (n, d) and `r` shape (n,): with `X` first, the inner numbers d and n differ, and Python stops."),
        draw.array_grids([("r", [[str(_v).replace("-", "−") for _v in C.R_SMALL]], ["shape (4,)"]),
                          ("X", [[str(_v) for _v in _row] for _row in C.X_SMALL], ["shape (4, 2)", "the first column"], {(0, 0), (1, 0), (2, 0), (3, 0)}),
                          ("r @ X", [["3", "4"]], ["shape (2,)", "the first number: 3"], {(0, 0)})],
                         title="One slope per column"),
        mo.md(r"""
        `r` times the first column of `X` is 2 × 1 + (−1) × 2 + 1 × 0 + 3 × 1 = 3: the first number of `r @ X`. The second column gives 4.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One step of gradient descent for linear regression:

    1. Predict for all examples at once: `y_pred = X @ w + b`.
    2. Work out the residuals: `r = y - y_pred`.
    3. Work out every slope with the chain rule: `grad_w = -2 * (r @ X) / n` and `grad_b = -2 * r.mean()`.
    4. Step against the slopes, `w = w - lr * grad_w` and `b = b - lr * grad_b`, and repeat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Eight houses, each with three features: the area in hundreds of m², a garden (1 = yes, 0 = no), and the distance to the station in km. The price is in thousands of euros. One step from $\mathbf{w} = 0$ and $b = 0$, with $\eta$ = 0.2, and a check of the first slope by measuring. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_step():
    import numpy as np

    _X = np.array(C.HOUSES)
    _y = np.array(C.PRICES)
    _w = np.zeros(3)
    _b = 0.0
    # 1. and 2. predict, and the residuals
    _r = _y - (_X @ _w + _b)
    # 3. every slope at once, with the chain rule
    _grad_w = -2 * (_r @ _X) / len(_y)
    _grad_b = -2 * _r.mean()
    print(_grad_w, _grad_b)

    # check the first slope by measuring, with a tiny h
    def _loss(w, b):
        return ((_y - (_X @ w + b)) ** 2).mean()

    _h = 0.001
    print((_loss(_w + np.array([_h, 0, 0]), _b) - _loss(_w, _b)) / _h)
    # 4. one update step
    _lr = 0.2
    print(_loss(_w, _b), _loss(_w - _lr * _grad_w, _b - _lr * _grad_b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The chain rule gives the slopes of about −394.7, −263 and −374 for the three weights, and −442.5 for $b$. Measuring the first one gives −394.66: the same; the chain rule needed no nudging at all. One step brings the loss down from 52310.5 to about 2530. In exercise 3 you repeat this step in a loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many slopes?
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "slope-shape",
        question="`X` holds 100 examples with 3 features: shape (100, 3). `r` holds their residuals: shape (100,). What is the shape of `r @ X`?",
        options={
            "right": "(3,)",
            "examples": "(100,)",
            "error": "No shape: they do not fit",
        },
        correct="right",
        why={
            "examples": "That would be one number per example. What does `r @ X` give for each column of `X`? See *The idea*.",
            "error": "Look at the inner numbers: (100,) and (100, 3). Are they equal? See *The idea*.",
            "right": "(100,) and (100, 3): the inner numbers, 100 and 100, match, and one number remains per column: one slope for each of the 3 weights.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Every slope at once

        Finish `mse_gradients(X, y, w, b)`. It gets the examples `X`, the labels `y`, the weights `w` and the bias term `b`, and gives back the slopes of the MSE: `grad_w`, one per weight, and `grad_b`. Replace the `...` with steps 1 to 3 of the recipe. The check calls `mse_gradients([[1, 2], [3, 0], [0, 1]], [3, 0, 3], [0, 0], 0)`: that should give `([-2.0, -6.0], -4.0)`. It also tries other examples, weights and bias terms. When your function is right, the check saves it in your project, as `work/projects/nn/gradients.py`. `return grad_w, grad_b` gives both back at once, as a pair.
        """),
        L.task("Make `mse_gradients` give back the slopes of the MSE for every weight and for `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("mse-gradients", lines=10)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("mse-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "mse-gradients",
        [
            "Steps 1 to 3 of the recipe, in the same order: the residuals first, then the two slopes.",
            "`r` is the label minus the prediction, for all examples at once. In which order do you multiply `r` and `X`?",
            "`r = y - (X @ w + b)`, `grad_w = -2 * (___) / len(y)` and `grad_b = -2 * ___`.",
        ],
        why="`r = y - (X @ w + b)` gives the residuals. `grad_w = -2 * (r @ X) / len(y)` is the mean of the chain-rule slopes for each weight, and `grad_b = -2 * r.mean()` the same for $b$. For the example, `r` is `[3, 0, 3]`, so `r @ X` is `[3, 9]`, and the slopes are −2 × [3, 9] / 3 = [−2, −6] and −2 × 2 = −4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Train

        Now your project goes to work. `fit(X, y, lr, steps)` imports your `mse_gradients` from your project, starts with all weights 0, one per column of `X` (`X.shape[1]` is the number of columns), and $b$ = 0, and runs its loop `steps` times. Replace the `...` with one update step: get both slopes with `gradients.mse_gradients(X, y, w, b)`, and move `w` and `b` against them. The check calls `fit([[1, 0], [0, 1], [1, 1]], [3, 2, 5], 0.1, 500)`: the three points fit $3x_1 + 2x_2$ exactly, so `w` should end near `[3, 2]` and `b` near 0. It also tries other data, learning rates and numbers of steps. When your function is right, the check saves `fit` in your project too, as `work/projects/nn/linear.py`.
        """),
        L.task("Make `fit` train `w` and `b` with your own `mse_gradients`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit", lines=13)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit, result_gradients):
    result_gradients  # check again when exercise 2 saves your module
    result_fit = L.check("fit", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit",
        [
            "Step 4 of the recipe, with the slopes from your own function.",
            "The function lives in the module `gradients`: call it through the module. It gives back two things at once: how do you give each its own name?",
            "`grad_w, grad_b = gradients.mse_gradients(X, y, w, b)`, then `w = w - ___` and `b = b - ___`.",
        ],
        why="`gradients.mse_gradients(X, y, w, b)` gives both slopes; `grad_w, grad_b = ...` catches them. `w = w - lr * grad_w` and `b = b - lr * grad_b` step against them. After 500 rounds on the example, `w` is about `[3.0, 2.0]` and `b` about 0.006.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your own `fit` to train a model on the eight houses, with $\eta$ = 0.2 and 1000 steps, and draws what it predicts against the real prices. Each blue point is a house; on the dashed line, a prediction is exactly right. It uses your own module, so it works once exercise 3 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_houses(result_fit):
    result_fit  # run again when exercise 3 saves your module
    _model = C.trained()
    if _model is None:
        _out = mo.md("*Finish exercise 3 first: then your `fit` is in your project, and this cell uses it.*")
    elif _model == "gradients":
        _out = mo.md("*Your `fit` gets its slopes from `mse_gradients` in `work/projects/nn/gradients.py`, and that file is missing, or no longer gives the right slopes. Check exercise 2 again; if you changed that file, delete it first.*")
    elif not _model:
        _out = mo.md("*The `fit` in `work/projects/nn/linear.py` no longer trains right, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*")
    else:
        _w, _b = _model
        _out = mo.vstack([
            mo.md(f"""Your model learned the weights <span class="c-param ea-num">{L.number(_w[0], 1)}</span> for the area, <span class="c-param ea-num">{L.number(_w[1], 1)}</span> for a garden and <span class="c-param ea-num">{L.number(_w[2], 1)}</span> for the distance, and the bias term <span class="c-param ea-num">{L.number(_b, 1)}</span>. So 100 m² more adds about 204 thousand euros, a garden about 17 thousand, and each km to the station costs about 39 thousand."""),
            mo.Html(C.demo_picture(_w, _b, {"title": "Your model on the eight houses", "x": "price (thousand euros)", "y": "prediction",
                                            "alt": "Eight points, one per house: the real price across, the prediction of your model up, from 140 to 350 thousand euros. All eight lie close to the dashed line, where a prediction is exactly right."})),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) you made every prediction at once; in [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) you found the slope of the loss of one example; in [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en) you stepped downhill. Together they are linear regression, and your project `nn` now has `gradients.py` and `linear.py`. In the next lesson, *Why gradient descent likes scaled features*, you see why the units of the houses were chosen with care: in m² and meters, $\eta$ = 0.2 blows up, and a learning rate that is small enough needs far more steps.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_fit, result_gradients):
    L.wrap_up(choice_shape, result_gradients, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
