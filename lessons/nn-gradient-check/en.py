# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-gradient-check. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-gradient-check", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en) and [The derivative as a slope](/course/open/math-slope?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grad():
    quick_grad = L.choice(
        "quick-grad",
        question="After `backward(loss)`, a weight `w` has `w.grad` = −3. What does that say?",
        options={
            "value": "`w` should become −3",
            "far": "`w` is 3 away from its best value",
            "right": "If `w` goes up a tiny bit, the loss goes down about 3 times as much",
        },
        correct="right",
        why={
            "value": "`grad` is not a new value for `w`, but the slope of the loss: how fast the loss changes when `w` changes. See [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en).",
            "far": "The slope says which way is downhill, and how steep it is there, not where the bottom is. See [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en).",
            "right": "`grad` is the slope of the loss for `w`: −3 means the loss falls about 3 for each unit that `w` goes up, as long as the step is small.",
        },
    )
    quick_grad
    return (quick_grad,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="The loss is 5.00 at $w = 2.00$ and 5.03 at $w = 2.01$. About what is its slope at $w = 2$?",
        options={
            "right": "3: the loss rises 0.03 over a step of 0.01",
            "small": "0.03: the loss rises by that much",
            "loss": "5.03: the loss after the step",
        },
        correct="right",
        why={
            "small": "0.03 is how much the loss rises, but the step was only 0.01: the slope is the rise divided by the step. See [The derivative as a slope](/course/open/math-slope?lang=en).",
            "loss": "The slope is not the loss itself, but how fast it changes. See [The derivative as a slope](/course/open/math-slope?lang=en).",
            "right": "Rise divided by step: 0.03 / 0.01 = 3. This lesson uses slopes measured in this way to check gradients.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en), your backward pass gave every parameter a `grad`, and training followed them. A mistake in a backward pass rarely crashes: a factor too many or a wrong sign still gives numbers, and the network still trains, only worse, or not at all, and nothing tells you why. Soon you write backward passes for whole layers. Before you trust one, you check it against a slope you can measure yourself, as you checked one gradient by moving a number in [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en).

    **An everyday example.** Before you trust a kitchen scale you built yourself, you weigh something whose weight you know, such as a litre of water.
    """)
    return


@app.cell(hide_code=True)
def predict_nudge():
    guess_nudge = L.guess(
        "up-and-down",
        question="You measure the slope of $f(x) = x^2$ at $x = 3$, which is really 6, in two ways, with $h = 0.1$. Which comes closer?",
        options={
            "up": "Only up: $(f(3.1) - f(3)) / 0.1$, because it starts exactly at 3",
            "same": "They are equally close, because both use the same $h$",
            "right": "Up and down: $(f(3.1) - f(2.9)) / 0.2$, because it looks at both sides",
        },
        reveal="**Up and down.** Only up gives $(9.61 - 9) / 0.1 = 6.1$; up and down gives $(9.61 - 8.41) / 0.2 = 6.0$, the slope itself. The curve bends: just right of 3 it is a little steeper, just left a little flatter. Up and down takes both sides, and their errors cancel; for a parabola even exactly.",
    )
    guess_nudge
    return (guess_nudge,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Gradient checking** compares the gradient from your backward pass with an estimate from nudging, one parameter at a time. In [The derivative as a slope](/course/open/math-slope?lang=en) you nudged up only: a **finite difference**. Nudge down as well and take the slope between the two, the **central difference**:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] \approx \frac{[[loss:\mathcal{L}(w_j + h)]] - [[loss:\mathcal{L}(w_j - h)]]}{2h}"),
        mo.md(r"""
        Its error shrinks with $h^2$, not with $h$: ten times smaller $h$, a hundred times smaller error. But $h$ cannot be too small: $\mathcal{L}(w_j + h)$ and $\mathcal{L}(w_j - h)$ then agree in almost all their digits, and their difference keeps little more than rounding. With NumPy's usual floats, $h = 10^{-5}$ is a good choice.

        Then compare the two with the **relative error**: the difference, divided by the larger of the two sizes, with $a$ the gradient from backprop and $n$ the estimate from nudging:
        """),
        L.formula(r"\text{relative error} = \frac{|a - n|}{\max(|a|, |n|)}"),
        mo.md(r"""
        Relative, because a gradient can be 1000 or 0.001: a difference of 0.001 says nothing on its own. Where both are 0, as for a ReLU that is off for every example, they agree: count it as 0. With $h = 10^{-5}$: below about $10^{-7}$ the two agree; above about $10^{-3}$ the backward pass is wrong for that parameter; in between, look closer. The check costs two runs of the loss for each parameter, so you check a small network, or a few parameters picked at random, once, after writing a backward pass; not during training.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **gradient checking** | | comparing the gradient from a backward pass with an estimate from nudging, for each parameter | |
        | **central difference** | | the slope between a nudge up and a nudge down: $(\mathcal{L}(w_j + h) - \mathcal{L}(w_j - h)) / 2h$ | `(f(w + step) - f(w - step)) / (2 * h)` |
        | $h$ | h | the size of the nudge; $10^{-5}$ is a good choice | `h = 1e-5` |
        | `np.abs` | | the size of every number, number by number: `np.abs(a - n)` drops the sign of each difference at once. Python's `abs(x)` works on one number only | `np.abs(a - n)` |
        | **relative error** | | the difference of two numbers divided by the larger of their sizes | `np.abs(a - n) / np.maximum(np.abs(a), np.abs(n))` |
        | `1e-5` | one e minus five | how Python writes $10^{-5}$: 0.00001. A number in front multiplies: `5.6e-12` is $5.6 \cdot 10^{-12}$ | `1e-5` |
        | `lambda w: ...` | lambda | a function in one line, without a name: `lambda w: np.sum(w ** 2)` gets `w` and gives back `np.sum(w ** 2)` | `lambda w: np.sum(w ** 2)` |
        """),
        L.watch_out("ReLU has a kink at 0. If a nudge takes an input of a ReLU across 0, the estimate mixes two slopes, and the check fails although your backward pass is right. When one parameter fails, look at whether an input sits near 0 before you hunt for a bug."),
        mo.md(r"""
        The picture: how far the two estimates of the slope of $e^x$ at $x = 1$ lie off, for each $h$. Going left, both get better, up and down much faster, until rounding wins: below about $10^{-5}$ for up and down and $10^{-8}$ for only up, the error grows again.
        """),
        mo.Html(C.h_picture({"title": "How far the estimate is off, for every h", "x": "step h", "y": "relative error",
                             "forward": "only up", "central": "up and down", "rounding": "rounding wins",
                             "alt": "Two lines on two log scales: the relative error against the step h, from 10 to the power minus 12 to 0.1. Up and down falls steeply to the right of 10 to the power minus 5 and is lowest near it, at about 10 to the power minus 11; only up, dashed, is lowest at about 10 to the power minus 8 near h = 10 to the power minus 8. Left of their lowest points both grow again, in a ragged line."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a backward pass:

    1. Work out the gradient with your backward pass: one number $a$ for each parameter.
    2. For each parameter: nudge it up by $h$ and down by $h$, work out the loss both times, and divide the difference by $2h$: the estimate $n$.
    3. Work out the relative error of each parameter: $|a - n| / \max(|a|, |n|)$.
    4. Judge: below about $10^{-7}$ they agree; above about $10^{-3}$, look for the bug in the backward pass of that parameter.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One neuron with two weights, a ReLU and a squared error, for the example $\mathbf{x} = (1, 2)$ with label 0.5. `_grad` is its backward pass, by hand with the chain rule. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then do what the comment says, and run it again.
    """)
    return


@app.cell
def example_check():
    import numpy as np

    _x = np.array([1.0, 2.0])
    _y = 0.5

    def _loss(w):
        return (np.maximum(w @ _x, 0) - _y) ** 2

    def _grad(w):
        # the backward pass: 2 · (relu(z) − y), times 1 where z > 0, times x
        _z = w @ _x
        return 2 * (np.maximum(_z, 0) - _y) * (_z > 0) * _x

    _w = np.array([0.3, 0.4])
    # 1. the gradient from the backward pass
    _a = _grad(_w)
    # 2. for each weight: nudge it up and down by h, and the slope between
    _h = 1e-5
    _n = np.zeros(2)
    for _j in range(2):
        _step = np.zeros(2)
        _step[_j] = _h
        _n[_j] = (_loss(_w + _step) - _loss(_w - _step)) / (2 * _h)
    print(_a, _n)
    # 3. the relative error of each weight
    print(np.abs(_a - _n) / np.maximum(np.abs(_a), np.abs(_n)))
    # 4. judge, and test the check itself: put a bug in _grad by leaving out the 2, and run again
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The backward pass and the nudges both give 1.2 and 2.4. The relative errors are about $6 \cdot 10^{-12}$ and $10^{-12}$ (Python prints `5.62...e-12` and `9.99...e-13`): far below $10^{-7}$, so the backward pass is right. With the 2 left out, the backward pass gives 0.6 and 1.2, the nudges still 1.2 and 2.4, and both relative errors are 0.5: far above $10^{-3}$. The check catches the bug, although the gradients still point the right way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bug():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Where is the bug?
    """)
    return


@app.cell(hide_code=True)
def choice_bug_cell():
    choice_bug = L.choice(
        "which-bug",
        question="A gradient check of four weights gives the relative errors $2 \\cdot 10^{-9}$, $4 \\cdot 10^{-10}$, $0.31$ and $10^{-8}$. What do you do?",
        options={
            "mean": "Nothing: on average the four errors are small enough",
            "right": "Look for a bug: in the backward pass of the third weight",
            "h": "Make $h$ smaller: 0.31 is rounding, and a smaller $h$ removes it",
        },
        correct="right",
        why={
            "mean": "One weight with a wrong gradient is enough to train badly, and the average hides it. Each parameter is judged on its own.",
            "h": "0.31 is not rounding: rounding gives errors far below $10^{-3}$, and a smaller $h$ makes rounding worse, not better.",
            "right": "Three are below $10^{-7}$, and 0.31 is far above $10^{-3}$: the backward pass is wrong for the third weight, unless an input of a ReLU sits near 0 there.",
        },
    )
    choice_bug
    return (choice_bug,)


@app.cell(hide_code=True)
def text_exercise_grad():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The estimate from nudging

        Write `numerical_grad(f, w, h)`: for each number of the array `w`, the central difference of the function `f`, by step 2 of the recipe, as an array. The check calls `numerical_grad(lambda w: np.sum(w ** 2), np.array([1.0, -2.0]), 0.001)`, which should give `[2, -4]`, and tries other functions, also ones in which the numbers of `w` work together, such as `np.sum(w) ** 2`.
        """),
        L.task("Make `numerical_grad` give the central difference of `f` for each number of `w`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grad():
    box_grad = L.code_box("numerical-grad", lines=10)
    box_grad
    return (box_grad,)


@app.cell(hide_code=True)
def check_grad(box_grad):
    result_grad = L.check("numerical-grad", box_grad.value)
    result_grad
    return (result_grad,)


@app.cell(hide_code=True)
def hints_grad():
    L.hints(
        "numerical-grad",
        [
            "As in step 2 of the worked example: a loop over `range(len(w))`, and for each number a nudge array of zeros with `h` in one place.",
            "`f(w + step)` and `f(w - step)` nudge only that one number, and leave `w` itself as it is. Divide their difference by `2 * h`, and collect the slopes in `np.zeros(len(w))`.",
        ],
        why="For `np.sum(w ** 2)` the slope of each number is twice that number: at `[1, -2]`, that is `[2, -4]`. The central difference gives it exactly, because the function is a parabola in each number.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_error():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The relative error

        Write `relative_error(a, n)`: for each parameter, the relative error between the gradient `a` from backprop and the estimate `n`, by step 3 of the recipe, as an array. The check calls `relative_error(np.array([2.0, -3.0, 0.4]), np.array([2.0, -4.0, 0.5]))`, which should give `[0, 0.25, 0.2]`, and tries large, small and negative gradients.
        """),
        L.task("Make `relative_error` give the relative error of each parameter."),
    ])
    return


@app.cell(hide_code=True)
def exercise_error():
    box_error = L.code_box("relative-error", lines=6)
    box_error
    return (box_error,)


@app.cell(hide_code=True)
def check_error(box_error):
    result_error = L.check("relative-error", box_error.value)
    result_error
    return (result_error,)


@app.cell(hide_code=True)
def hints_error():
    L.hints(
        "relative-error",
        [
            "The size of a number is its value without sign: `np.abs`. Work on whole arrays at once, as `np.abs(a - n)`.",
            "`np.maximum(x, y)` takes the larger of two arrays, place by place, as in a ReLU.",
        ],
        why="For the second parameter: $|-3 - (-4)| = 1$, and the larger size is 4: $1 / 4 = 0.25$. For the third: $0.1 / 0.5 = 0.2$.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The slope you measured by nudging in [The derivative as a slope](/course/open/math-slope?lang=en), taken on both sides, becomes a test for the backward pass of [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en): two runs of the loss for each parameter, and a relative error that says whether a gradient is right. In the next lesson, *Backprop through whole layers by hand*, you write the backward pass of a whole network in NumPy, and this check, and PyTorch, tell you whether it is right.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bug, result_grad, result_error):
    L.wrap_up(choice_bug, result_grad, result_error)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
