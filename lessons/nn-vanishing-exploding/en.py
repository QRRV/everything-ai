# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-vanishing-exploding. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-vanishing-exploding", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) and [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sigmoid():
    quick_sigmoid = L.choice(
        "quick-sigmoid",
        question="Roughly what does the sigmoid give for $z = 10$, and for $z = -10$?",
        options={
            "right": "Almost 1, and almost 0: it flattens out on both sides",
            "same": "10 and −10: a large score stays large",
            "half": "0.5 for both: the sigmoid is symmetric",
        },
        correct="right",
        why={
            "same": "The sigmoid squeezes every score into a probability between 0 and 1. See [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "half": "The sigmoid gives 0.5 only at $z = 0$; the further from 0, the closer to 1 or 0. See [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "right": "$e^{-10}$ is tiny, so $\\sigma(10) \\approx 0.99995$; and $\\sigma(-10) \\approx 0.00005$. Far from 0 the curve is almost flat: this lesson needs that.",
        },
    )
    quick_sigmoid
    return (quick_sigmoid,)


@app.cell(hide_code=True)
def quick_check_keep():
    quick_keep = L.choice(
        "quick-keep",
        question="Why does a backward pass keep the `Z1` of the forward pass, the inputs of the ReLU?",
        options={
            "weights": "To change the weights: gradient descent needs it",
            "loss": "To work out the loss again: the backward pass starts there",
            "right": "For the slope of the ReLU: 1 where `Z1` is above 0, else 0",
        },
        correct="right",
        why={
            "weights": "The weights change with their gradients, `dW1` and `dW2`. See [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en).",
            "loss": "The loss comes from `P`, at the end of the forward pass. See [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en).",
            "right": "Going back through the ReLU, the gradient is multiplied by its slope at `Z1`: `dH * (Z1 > 0)`. Every layer multiplies by such a slope, and this lesson is about what that does over many layers.",
        },
    )
    quick_keep
    return (quick_keep,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A deeper network can learn more, so why not stack fifty layers? For a long time, that did not work: the first layers of a deep network hardly changed, or training stopped with a loss of `nan`. The reason is in the backward pass of [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en): on its way back, the gradient is multiplied once by every layer. Knowing which, and why, is the first step to training deep networks at all.
    """)
    return


@app.cell(hide_code=True)
def predict_chain():
    guess_chain = L.guess(
        "ten-sigmoids",
        question="A chain of 10 layers, each one sigmoid with weight 1, with the input 0.5. The gradient at the top is 1. How large do you think it is when it comes back to the input?",
        options={
            "quarter": "About 0.25: the gradient goes back through one sigmoid at a time",
            "hundredth": "About 0.01: it shrinks a little in each layer",
            "right": "Less than a millionth: it shrinks a lot in each layer",
        },
        reveal="**Less than a millionth.** The largest slope of the sigmoid is 0.25, at $z = 0$. Each of the 10 layers multiplies the gradient by at most 0.25: $0.25^{10} \\approx 0.00000095$. Here the slopes are a little below 0.25, and what comes back is about 0.00000035. The first layer of such a network hardly learns.",
    )
    guess_chain
    return (guess_chain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Take a chain of $n$ layers with one number each: layer $k$ gives $h_k = f(z_k)$, with $z_k = w_k h_{k-1}$, $h_0$ the input and $f$ the activation function. By the chain rule, the gradient that comes back to the input is the gradient at the top times one **factor** per layer: its weight times the slope of its activation function at its $z$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial h_0}]] = [[grad:\frac{\partial \mathcal{L}}{\partial h_n}]] \cdot [[param:w_n]] f'(z_n) \cdot \ldots \cdot [[param:w_2]] f'(z_2) \cdot [[param:w_1]] f'(z_1)"),
        mo.md(r"""
        Many factors below 1 make the product tiny: the gradient **vanishes**, and the first layers hardly learn. Many factors above 1 make it huge: the gradient **explodes**, the steps of gradient descent shoot off, and the loss jumps up or becomes `inf` or `nan`.

        The slope of the sigmoid, as in [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en), is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. It is at most 0.25, at $z = 0$, and almost 0 far from it, where the sigmoid is almost flat: there the sigmoid is **saturated**, and it lets almost no gradient through. So in a chain of sigmoids the gradient vanishes, even with larger weights: those push $z$ away from 0, into the flat parts.

        The slope of ReLU is 1 where $z > 0$, and 0 elsewhere: what it lets through keeps its size, without a factor such as the 0.25 of the sigmoid. Then the weights decide. With matrices, each layer multiplies the gradient by its matrix, as `(g * (z > 0)) @ W`: weights that are too small shrink it, layer after layer; too large, and it grows.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **vanishing gradient** | | a gradient that shrinks layer after layer on the way back, until the first layers hardly learn | |
        | **exploding gradient** | | a gradient that grows layer after layer on the way back, until the steps shoot off | |
        | **saturation** | | where an activation function such as the sigmoid is almost flat, so its slope, and the gradient it lets through, is almost 0 | |
        | $\sigma'(z)$ | sigma prime of z | the slope of the sigmoid at $z$: $\sigma(z)(1 - \sigma(z))$, at most 0.25 | `sigmoid(z) * (1 - sigmoid(z))` |
        """),
        L.watch_out("The loss alone does not tell you which of the two you have. Vanishing looks like a loss that hardly moves; exploding like a loss that jumps or becomes `nan`, which a too large learning rate can do as well. Look at the size of the gradient per layer: tiny in the first layers means vanishing, huge means exploding."),
        mo.md(r"""
        The picture: the mean size of the gradient on its way back through 20 ReLU layers of 50 neurons each, for three spreads of random starting weights. Too small, and it shrinks tenfold about every 8 layers; too large, and it grows; in between it stays about the same size. Finding that spread is the next lesson.
        """),
        mo.Html(C.sizes_picture({"title": "The gradient on its way back through 20 ReLU layers", "x": "layers back from the top",
                                 "y": "mean size of the gradient", "big": "weights too large", "right": "just right", "small": "weights too small",
                                 "alt": "Three lines on a log scale over 20 layers. With weights too large the size grows from about 1.5 to about 2,600; with weights just right, dashed, it stays between about 1 and 4; with weights too small it falls from about 0.8 to about 0.003."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To follow the gradient back through a chain of layers:

    1. The forward pass, keeping each $z$.
    2. Start at the top with the gradient there; here 1.
    3. Go back through the layers, from the last to the first, and multiply by the factor of each: its weight times the slope of its activation function at its $z$.
    4. Compare what comes back with what you started with: much smaller means vanishing, much larger exploding.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The chain of the guess: 10 sigmoid layers, each with weight 1, and the input 0.5. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again. `np.round(_zs, 3)` shows the $z$'s rounded to 3 decimals.
    """)
    return


@app.cell
def example_chain():
    import numpy as np

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    _w = 1.0
    _h = 0.5
    _zs = []
    # 1. the forward pass through 10 layers, keeping each z
    for _k in range(10):
        _z = _w * _h
        _zs.append(_z)
        _h = _sigmoid(_z)
    print(np.round(_zs, 3))
    # 2. the gradient at the top
    _g = 1.0
    # 3. back from the last layer to the first: times the weight and the slope of the sigmoid
    for _z in reversed(_zs):
        _g = _g * _w * _sigmoid(_z) * (1 - _sigmoid(_z))
        print(_g)
    # 4. compare: print how many times smaller than the 1 at the top it is, 1 / _g, and run again;
    # then: do larger weights help? Set _w to 4.0 and run once more
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    After a few layers every $z$ is about 0.66, where the slope of the sigmoid is about 0.22: each layer multiplies the gradient by that, from 0.225 to 0.0505 and on, to about 0.00000035 at the input: about 2.9 million times smaller than at the top. With `_w = 4.0` it is worse: after the first two layers the $z$'s are about 3.9, where the sigmoid is saturated and its slope about 0.019; even times 4, each of those layers multiplies by only about 0.076, and about 0.00000000005 comes back, about 18 billion times smaller. Larger weights push a sigmoid into its flat parts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nan():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A loss of nan
    """)
    return


@app.cell(hide_code=True)
def choice_nan_cell():
    choice_nan = L.choice(
        "nan-loss",
        question="You train a ReLU network of 30 layers. After a few steps the loss jumps up, and then it is `nan`. The learning rate worked well for a network of 3 layers. What is the most likely cause?",
        options={
            "vanish": "Vanishing gradients: the first layers stop learning altogether",
            "right": "Exploding gradients: the steps of the first layers shoot off",
            "data": "A `nan` in the training data, in one of the images",
        },
        correct="right",
        why={
            "vanish": "Vanishing gradients make the loss hardly move; they do not make it jump or become `nan`.",
            "data": "A `nan` in the data would give a loss of `nan` from the first steps on, also with 3 layers, without jumping up first.",
            "right": "With weights that stretch the gradient, 30 factors above 1 make it huge, and the steps with it; the loss jumps, and then a number overflows to `inf` and `nan`. With 3 layers there were too few factors to notice.",
        },
    )
    choice_nan
    return (choice_nan,)


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Back through a chain of sigmoids

        Write `chain_gradient(ws, x)`: a chain of sigmoid layers, one for each weight in the list `ws`: each layer gives `sigmoid(w * h)`, with `h` what the layer before gave, starting with `x`. Give back the gradient that arrives at `x`, starting with 1 at the top, by the recipe. `sigmoid` is already in the box. The check calls `chain_gradient([2.0, 1.0, 1.5], 0.5)`, which should give about `0.02529`, and tries other weights and more layers.
        """),
        L.task("Make `chain_gradient` give the gradient that arrives at `x` through the chain."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain-gradient", lines=14)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("chain-gradient", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "chain-gradient",
        [
            "As in the worked example, but with the weight of each layer from `ws`: in the forward pass, keep each `z` in a list.",
            "On the way back, go through the layers last first, `for k in reversed(range(len(ws))):`, with the weight `ws[k]` and the `zs[k]` of each.",
        ],
        why="With the weights 2, 1 and 1.5 and the input 0.5, the $z$'s are 1, 0.731 and 1.013. The factor of each layer, its weight times the slope of the sigmoid at its $z$, is $2 \\times 0.197$, $1 \\times 0.219$ and $1.5 \\times 0.195$: 0.393, 0.219 and 0.293, and their product is about 0.025.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_relu():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The size of the gradient per layer

        Write `relu_grad_sizes(Ws, x)`: a ReLU network with one weight matrix per layer in the list `Ws`, where each layer gives `np.maximum(h @ W.T, 0)`, starting with the example `x`. Start the way back with a gradient of ones at the top, `np.ones(len(h))`: like `np.zeros`, but with ones. Going back through a layer, keep the gradient where its `z` was above 0, then multiply it by `W`, as in [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en). After each layer, add the mean size of the gradient, `np.mean(np.abs(g))`, to a list, and give the list back, from the top layer to the first. The check calls `relu_grad_sizes([np.array([[1.0, -1.0], [0.5, 2.0]]), np.array([[2.0, 0.0], [-1.0, 1.0]])], np.array([1.0, -1.0]))`, which should give `[1, 2]`, and tries deeper and wider networks.
        """),
        L.task("Make `relu_grad_sizes` give the size of the gradient after each layer, on the way back."),
    ])
    return


@app.cell(hide_code=True)
def exercise_relu():
    box_relu = L.code_box("relu-sizes", lines=14)
    box_relu
    return (box_relu,)


@app.cell(hide_code=True)
def check_relu(box_relu):
    result_relu = L.check("relu-sizes", box_relu.value)
    result_relu
    return (result_relu,)


@app.cell(hide_code=True)
def hints_relu():
    L.hints(
        "relu-sizes",
        [
            "The forward pass keeps each `z`, as in exercise 2, but now `z = h @ W.T` and `h = np.maximum(z, 0)`. The way back starts with `g = np.ones(len(h))`.",
            "Going back, from the last layer to the first, with `for k in reversed(range(len(Ws))):`, the new gradient is `g = (g * (zs[k] > 0)) @ Ws[k]`; then put the size of this `g` in the list.",
        ],
        why="Forward: $z_1 = (2, -1.5)$, $h_1 = (2, 0)$, $z_2 = (4, -2)$. Back: the ones kept where $z_2 > 0$ are $(1, 0)$, times the second $W$ gives $(2, 0)$, mean size 1; kept where $z_1 > 0$, $(2, 0)$, times the first $W$ gives $(2, -2)$, mean size 2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The backward pass of [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en) multiplies the gradient by every layer once; with many layers those factors decide everything. The flat ends of [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) make it vanish; weights with the wrong spread make it vanish or explode. In the next lesson, *Initialization*, you work out the spread that keeps the signal the same size, layer after layer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_nan, result_chain, result_relu):
    L.wrap_up(choice_nan, result_chain, result_relu)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
