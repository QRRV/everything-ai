# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-backprop-tensors. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-backprop-tensors", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) and [Gradient checking](/course/open/nn-gradient-check?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_autograd():
    quick_autograd = L.choice(
        "quick-autograd",
        question="`w = torch.tensor([1.0, -2.0], requires_grad=True)`, `loss = (w * 3).sum()` and `loss.backward()`. What is `w.grad`?",
        options={
            "right": "`tensor([3., 3.])`: each number of `w` counts three times in the sum",
            "times": "`tensor([3., -6.])`: each number of `w`, times the 3 it is multiplied by in the loss",
            "one": "`tensor(-3.)`: one gradient for the whole sum",
        },
        correct="right",
        why={
            "times": "That is `w * 3` itself. The gradient says how fast the loss changes when one number of `w` goes up a little: by 3, whatever that number is. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "one": "`w.grad` has the shape of `w`: one gradient for each of its numbers. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "right": "If one number of `w` goes up by a little, the loss goes up by three times that little: 3 for each, in the shape of `w`.",
        },
    )
    quick_autograd
    return (quick_autograd,)


@app.cell(hide_code=True)
def quick_check_error():
    quick_error = L.choice(
        "quick-error",
        question="Your backward pass gives a weight the gradient 0.8; nudging gives 0.4. What is the relative error?",
        options={
            "small": "1: the difference, 0.4, divided by the smaller of the two",
            "abs": "0.4: the difference itself",
            "right": "0.5: the difference, 0.4, divided by the larger of the two",
        },
        correct="right",
        why={
            "small": "The relative error divides by the larger of the two sizes. See [Gradient checking](/course/open/nn-gradient-check?lang=en).",
            "abs": "That is the difference; the relative error divides it by the larger size, so that large and small gradients are judged alike. See [Gradient checking](/course/open/nn-gradient-check?lang=en).",
            "right": "0.4 / 0.8 = 0.5: far above $10^{-3}$, so that backward pass has a bug. In this lesson your own backward pass should come out far below $10^{-7}$.",
        },
    )
    quick_error
    return (quick_error,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your `Value` from [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en) did backprop one number at a time. For the MLP that reads the digits, with 64 hidden neurons and a mini-batch of 64 images, that is millions of `Value`s in every step: far too slow. PyTorch does the same backward pass a whole layer at a time, with a few matrix products. Today you write that backward pass yourself, in NumPy, for a two-layer MLP with softmax and cross-entropy, and you check it against PyTorch. After that, `loss.backward()` holds no secrets: it does these products.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           8, here_label="← today", title="Where your project stands: your own neural network"),
    ])
    return


@app.cell(hide_code=True)
def predict_relu():
    guess_relu = L.guess(
        "through-relu",
        question="Going back, the gradient for the output of a ReLU arrives: one number per example and per neuron. What do you think the gradient for its input is?",
        options={
            "same": "The same numbers: ReLU has no weights, so it passes the gradient on unchanged",
            "right": "The same numbers where the input was above 0, and 0 elsewhere",
            "times": "The numbers times what the ReLU gave out",
        },
        reveal="**The same numbers where the input was above 0, and 0 elsewhere.** ReLU lets a positive input through unchanged, so a little more input gives a little more output: slope 1. A negative input gives 0, and a little more of it still gives 0: slope 0. So going back, the gradient passes where the input was above 0, and stops elsewhere, as the local gradient of `relu` in your `Value` did, now for a whole matrix at once.",
    )
    guess_relu
    return (guess_relu,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The MLP of the digits, for a mini-batch of $n$ images $\mathbf{X}$, as in [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en), with softmax and the cross-entropy loss at the end:
        """),
        L.formula(r"\begin{aligned} \mathbf{Z}_1 &= \mathbf{X}[[param:\mathbf{W}_1]]^\top + [[param:\mathbf{b}_1]], \quad \mathbf{H} = \mathrm{ReLU}(\mathbf{Z}_1) \\ \mathbf{Z}_2 &= \mathbf{H}[[param:\mathbf{W}_2]]^\top + [[param:\mathbf{b}_2]], \quad [[pred:\mathbf{P}]] = \mathrm{softmax}(\mathbf{Z}_2), \quad [[loss:\mathcal{L}]] = -\frac{1}{n}\sum_i \log [[pred:P_{i,c_i}]] \end{aligned}"),
        mo.md(r"""
        Here $c_i$ is the right class of example $i$. The backward pass goes the other way, and each step is a rule you know. $\mathbf{G}$ is the gradient of the loss for what a layer gives out, with the same shape:
        """),
        L.formula(r"\begin{aligned} [[grad:\mathbf{G}_2]] &= ([[pred:\mathbf{P}]] - \mathbf{Y}) / n \\ [[grad:d\mathbf{W}_2]] &= [[grad:\mathbf{G}_2]]^\top \mathbf{H}, \quad [[grad:d\mathbf{b}_2]] = \textstyle\sum_i [[grad:\mathbf{G}_2]], \quad [[grad:d\mathbf{H}]] = [[grad:\mathbf{G}_2]][[param:\mathbf{W}_2]] \\ [[grad:\mathbf{G}_1]] &= [[grad:d\mathbf{H}]] \odot [\mathbf{Z}_1 > 0] \\ [[grad:d\mathbf{W}_1]] &= [[grad:\mathbf{G}_1]]^\top \mathbf{X}, \quad [[grad:d\mathbf{b}_1]] = \textstyle\sum_i [[grad:\mathbf{G}_1]] \end{aligned}"),
        mo.md(r"""
        - The first line is the gradient of softmax with cross-entropy, as in [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en), for a whole mini-batch: $\mathbf{Y}$ holds the one-hot rows of the labels, and the $1/n$ comes from the mean.
        - $d\mathbf{W}$ and $d\mathbf{H}$ are the two products of [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en): for $\mathbf{X}\mathbf{W}^\top$, the weights get $\mathbf{G}^\top\mathbf{X}$ and the input gets $\mathbf{G}\mathbf{W}$.
        - A bias term is added to every example, so it gets the gradient of each, added up over the examples: one number per output.
        - ReLU passes the gradient where its input was above 0, and stops it elsewhere; the element-wise product $\odot$ multiplies place by place.

        So going back, each layer needs what it got going forward: the backward pass keeps $\mathbf{X}$, $\mathbf{Z}_1$, $\mathbf{H}$ and $\mathbf{P}$ from the forward pass. Your linear classifier of [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en) had the first line, $d\mathbf{W}_2$ and $d\mathbf{b}_2$ already, with $\mathbf{X}$ where $\mathbf{H}$ is now.

        | | Your `Value` | A whole layer |
        |---|---|---|
        | what one step holds | one number | a matrix, one row per example |
        | the local gradient of a product | the other number | the other matrix, in a product: $\mathbf{G}^\top\mathbf{X}$ and $\mathbf{G}\mathbf{W}$ |
        | ReLU | 1 or 0 | the matrix $[\mathbf{Z}_1 > 0]$ |
        | a number used more than once | gradients added up | the sum over the examples, $\sum_i$ |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | $\mathbf{G}$ | G | the gradient of the loss for what a layer gives out, in its shape: one row per example | `G2` |
        | $\mathbf{Y}$ | Y | the one-hot rows of the labels: a 1 at the right class of each example | `np.eye(10)[y]` |
        | **element-wise product**, $\odot$ | element-wise product | multiplying two matrices of the same shape number by number | `dH * (Z1 > 0)` |
        | $[\mathbf{Z}_1 > 0]$ | Z one above zero | 1 where $\mathbf{Z}_1$ is above 0, else 0 | `(Z1 > 0)` |
        | $\sum_i$ | sum over i | added up over the examples, one number per column | `G2.sum(axis=0)` |
        """),
        L.watch_out("Keep the shapes in sight: every gradient has the shape of what it belongs to. A product the wrong way round, such as `X.T @ G`, still runs, and gives the gradient of the weights turned over; when `W` is square, even its shape looks right, and nothing stops you. And keep `Z1` from the forward pass: the slope of the ReLU depends on it."),
        mo.md(r"""
        The picture: the forward pass from left to right, with the shape of each step, and the backward pass under it, from right to left.
        """),
        draw.flow([("X", ["images", "(n, 784)"]), ("H", ["after ReLU", "(n, 64)"]), ("P", ["probabilities", "(n, 10)"]), ("Loss", ["one number"])],
                  ["W1, b1, ReLU", "W2, b2, softmax", "mean of\n−log P"],
                  ["G1 = dH ⊙ [Z1 > 0]\ndW1 = G1ᵀ X\ndb1 = Σ G1", "dW2 = G2ᵀ H\ndb2 = Σ G2\ndH = G2 W2", "G2 = (P − Y) / n"],
                  title="The forward pass, and the backward pass under it"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To work out the gradients of a two-layer MLP with softmax and cross-entropy:

    1. The forward pass, keeping `Z1`, `H` and `P`.
    2. The gradient for the logits, `G2 = (P - Y) / n`, and the output layer: `dW2 = G2.T @ H` and `db2 = G2.sum(axis=0)`.
    3. Back through the output layer and the ReLU: `dH = G2 @ W2`, then `G1 = dH * (Z1 > 0)`.
    4. The hidden layer: `dW1 = G1.T @ X` and `db1 = G1.sum(axis=0)`.
    5. Check against PyTorch: the same forward pass with tensors that have `requires_grad=True`, then `loss.backward()`, and the relative error between each gradient and its `.grad`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small MLP: 3 examples with 2 features, 2 hidden neurons and 3 classes. The comments with `#` say which step of the recipe each part is. Step 4 is not finished, and neither is its line in step 5: run the cell as it is, then complete both where the comments say, and run it again. In step 5, `torch.exp` and `torch.log` work on a tensor as `np.exp` and `np.log` do on an array, and `.numpy()` turns a tensor back into an array.
    """)
    return


@app.cell
def example_backprop():
    import numpy as np
    import torch

    _X = np.array([[1.0, 0.5], [0.5, 1.0], [0.0, 2.0]])
    _y = np.array([0, 2, 1])
    _W1 = np.array([[1.0, -0.6], [0.5, 0.5]])
    _b1 = np.array([0.0, -0.4])
    _W2 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
    _b2 = np.array([0.1, 0.0, -0.1])
    _n = len(_y)

    # 1. the forward pass, keeping Z1, H and P
    _Z1 = _X @ _W1.T + _b1
    _H = np.maximum(_Z1, 0)
    _Z2 = _H @ _W2.T + _b2
    _E = np.exp(_Z2 - _Z2.max(axis=1, keepdims=True))
    _P = _E / _E.sum(axis=1, keepdims=True)
    # 2. the gradient for the logits, and the output layer
    _G2 = (_P - np.eye(3)[_y]) / _n
    _dW2 = _G2.T @ _H
    _db2 = _G2.sum(axis=0)
    # 3. back through the output layer and the ReLU
    _dH = _G2 @ _W2
    _G1 = _dH * (_Z1 > 0)
    # 4. the hidden layer: dW1 is here; add db1, the way db2 was made, and print it
    _dW1 = _G1.T @ _X
    print(_dW1)

    # 5. the same in PyTorch, and the relative error of each gradient
    _tW1 = torch.tensor(_W1, requires_grad=True)
    _tb1 = torch.tensor(_b1, requires_grad=True)
    _tW2 = torch.tensor(_W2, requires_grad=True)
    _tb2 = torch.tensor(_b2, requires_grad=True)
    _tZ2 = torch.relu(torch.tensor(_X) @ _tW1.T + _tb1) @ _tW2.T + _tb2
    _tE = torch.exp(_tZ2)
    _tP = _tE / _tE.sum(dim=1, keepdim=True)
    _loss = -(torch.tensor(np.eye(3)[_y]) * torch.log(_tP)).sum(dim=1).mean()
    _loss.backward()

    def _rel(a, n):
        return np.max(np.abs(a - n) / np.maximum(np.abs(a), np.abs(n)))

    print(_rel(_dW1, _tW1.grad.numpy()), _rel(_dW2, _tW2.grad.numpy()), _rel(_db2, _tb2.grad.numpy()))
    # and the line for db1: print(_rel(_db1, _tb1.grad.numpy()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first example has both hidden inputs above 0; the second and third have their first one below 0, so `G1` is 0 there, and only the first example reaches the first row of `dW1`: 0.0223 and 0.0111. The three relative errors are at most a few times $10^{-15}$: the same numbers as PyTorch, up to rounding. With step 4 complete, `db1` is 0.0223 and 0.1804, and its relative error is just as small. PyTorch's `loss.backward()` did these same products.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The gradient of a bias term
    """)
    return


@app.cell(hide_code=True)
def choice_bias_cell():
    choice_bias = L.choice(
        "bias-sum",
        question="Why is `db2` the sum of `G2` over the examples, `G2.sum(axis=0)`, and not its mean?",
        options={
            "right": "`b2` is added to every example, so its gradients add up; the $1/n$ is already in `G2`",
            "mean": "It should be the mean: the loss is the mean over the examples, so the gradient of `b2` is a mean as well",
            "axis": "The sum is over the classes, so that each example gets one number",
        },
        correct="right",
        why={
            "mean": "The loss is a mean, but that mean is already in `G2 = (P - Y) / n`. Taking the mean again divides by $n$ twice.",
            "axis": "`axis=0` adds up over the examples, the rows, and leaves one number per class: the shape of `b2`.",
            "right": "`b2` is used once for every example, and a number used more than once gets its gradients added up, as in your `Value`. The $1/n$ is already in `G2`.",
        },
    )
    choice_bias
    return (choice_bias,)


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One linear layer

        Write `linear_backward(G, X, W)`: the backward pass of one linear layer `X @ W.T + b`. It gets `G`, the gradient for what the layer gave out, of shape (examples, outputs), the input `X`, of shape (examples, inputs), and the weights `W`, of shape (outputs, inputs), and gives back `dX`, `dW` and `db`, in that order. The check calls it with `G` of shape (3, 2), `X` of shape (3, 4) and `W` of shape (2, 4), and with other shapes. When your function is right, the check saves it in your project, as `work/projects/nn/layer_backward.py`.
        """),
        L.task("Make `linear_backward` give the gradients for the input, the weights and the bias terms of a linear layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("linear-backward", lines=8)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("linear-backward", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "linear-backward",
        [
            "Steps 2 to 4 of the recipe do this for each layer: one product for the weights, one for the input, and a sum for the bias terms.",
            "Let the shapes decide each product: `dW` must be (outputs, inputs), `dX` (examples, inputs), and `db` one number per output.",
        ],
        why="`dX = G @ W` is (examples, outputs) times (outputs, inputs); `dW = G.T @ X` is (outputs, examples) times (examples, inputs); `db = G.sum(axis=0)` adds up over the examples. Together: `return dX, dW, db`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The whole MLP

        Write `mlp_gradients(X, y, W1, b1, W2, b2)`: the gradients of the cross-entropy loss of the MLP for `W1`, `b1`, `W2` and `b2`, in that order, by steps 1 to 4 of the recipe. `X` has one row per example, `y` the right class of each, as a number; the number of classes is `W2.shape[0]`. You may copy your `linear_backward` into the box and use it. The check calls it on 4 examples with 2 features, 3 hidden neurons and 2 classes, and looks at each gradient; then on a larger MLP with 3 classes. When your function is right, the check saves it in your project, as `work/projects/nn/mlp_backward.py`.
        """),
        L.task("Make `mlp_gradients` give the four gradients of the MLP."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("mlp-gradients", lines=18)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("mlp-gradients", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "mlp-gradients",
        [
            "As in the worked example, from top to bottom: the forward pass, then steps 2, 3 and 4. `np.eye(W2.shape[0])[y]` gives the one-hot rows.",
            "Give back the four gradients in the order of `W1, b1, W2, b2` on the `def` line: `return dW1, db1, dW2, db2`.",
        ],
        why="The forward pass gives `Z1`, `H` and `P`. Then `G2 = (P - np.eye(W2.shape[0])[y]) / len(y)`, `dW2 = G2.T @ H`, `db2 = G2.sum(axis=0)`, `G1 = (G2 @ W2) * (Z1 > 0)`, `dW1 = G1.T @ X` and `db1 = G1.sum(axis=0)`, and `return dW1, db1, dW2, db2`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below takes your own `mlp_gradients` from your project. First it compares its gradients with those of PyTorch, for an MLP with 64 hidden neurons on 64 digits. Then it trains that MLP with them on the 6,000 training digits, in NumPy only, for 5 epochs of mini-batches of 64 with $\eta$ = 0.5, and measures it on the 1,000 test digits. That takes a few seconds. It works once exercise 3 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_backprop(result_mlp):
    result_mlp  # run again when exercise 3 saves your function
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercise 3 first: then your `mlp_gradients` is in your project, and this cell uses it.*")
    elif _result == "slow":
        _out = mo.md("*Training with your `mlp_gradients` in `work/projects/nn/mlp_backward.py` takes more than a minute. Does it use whole matrices, without a loop over the examples?*")
    elif _result == "broken":
        _out = mo.md("*The `mlp_gradients` in `work/projects/nn/mlp_backward.py` stops with an error, or gives other gradients than PyTorch. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*")
    else:
        _errors, _accuracy, _losses = _result
        _out = mo.vstack([
            mo.md(f"""The largest relative error of your four gradients against PyTorch is below $10^{{{C.power(max(_errors))}}}$: far below $10^{{-7}}$, so your backward pass computes what `loss.backward()` computes. Trained with it, your MLP reads <span class="ea-num">{L.number(round(_accuracy * 1000), 0)}</span> of the 1,000 test digits right: an accuracy of <span class="ea-num">{L.number(_accuracy, 3)}</span>, where your linear classifier read 890; and without a single line of PyTorch."""),
            mo.Html(C.losses_picture(_losses, {"title": "The loss of your training, per epoch", "epoch": "epoch", "loss": "loss on the training digits",
                                               "alt": f"A line through {len(_losses)} points, one per epoch: the loss on all training digits falls from {L.number(_losses[0], 2)} after the first epoch to {L.number(_losses[-1], 2)} after the last."})),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The two products of [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en), the gradient of [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en) and the ReLU of your `Value` together make the backward pass of a whole MLP; [Gradient checking](/course/open/nn-gradient-check?lang=en) and PyTorch confirm it. Your project `nn` now has `layer_backward.py` and `mlp_backward.py` as well. In the next lesson, *Vanishing and exploding gradients*, you follow these gradients back through many layers, and see them shrink or grow.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bias, result_layer, result_mlp):
    L.wrap_up(choice_bias, result_layer, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
