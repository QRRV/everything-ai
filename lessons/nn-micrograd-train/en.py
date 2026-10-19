# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-micrograd-train. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-train", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Neurons, layers and an MLP on top of your Value](/course/open/nn-micrograd-mlp?lang=en) and [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_output():
    quick_output = L.choice(
        "quick-output",
        question="`m = MLP(2, [3, 1])`: 2 inputs, a hidden layer of 3 neurons, and one output neuron. What does `m([Value(1), Value(2)])` give back?",
        options={
            "right": "A list with one `Value`",
            "hidden": "A list with 3 `Value`s, one per hidden neuron",
            "one": "One `Value`, without a list around it",
        },
        correct="right",
        why={
            "hidden": "That is what the hidden layer gives. The MLP passes it on to the next layer, and gives back what its last layer gives. See [Neurons, layers and an MLP on top of your Value](/course/open/nn-micrograd-mlp?lang=en).",
            "one": "A layer gives back a list, one `Value` per neuron, also when it has only one neuron, and the MLP gives back what its last layer gives. See [Neurons, layers and an MLP on top of your Value](/course/open/nn-micrograd-mlp?lang=en).",
            "right": "The MLP gives back what its last layer gives: a list with one `Value` per neuron. With one output neuron, that is a list with one `Value`, and `[0]` takes it out.",
        },
    )
    quick_output
    return (quick_output,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_slopes = L.choice(
        "quick-slopes",
        question="One step of gradient descent with two parameters, $w$ and $b$. Which slopes does it use?",
        options={
            "after": "The slope of $w$; then, once $w$ has moved, the slope of $b$ at the new $w$",
            "larger": "Only the larger of the two slopes; the other parameter waits its turn",
            "right": "Both slopes, measured at the same $w$ and $b$, before either moves",
        },
        correct="right",
        why={
            "after": "Then the slope of $b$ belongs to another point than the slope of $w$. See the recipe in [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en).",
            "larger": "Each parameter moves against its own slope, in every step. See [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en).",
            "right": "Both slopes belong to the same point, so together they point the way down from there. Only then do $w$ and $b$ both move.",
        },
    )
    quick_slopes
    return (quick_slopes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your project can now build an MLP of any size and give each of its parameters a slope. What it cannot do yet is learn. Learning is a loop: predict, measure the loss, get the slopes, take a small step against them, and again. That loop trains every network in this course, up to the language model at the end, and PyTorch writes it in the same steps. Today you write it, in your project. That is the seventh step of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           6, here_label="← today", title="Where your project stands: your own neural network"),
        L.course_versions(),
        mo.md(r"""
        **An everyday example.** Learning to ride a bike: you push off, wobble, lean a little the other way, and try again. Each try needs only a small correction; many of them make you ride.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_pile():
    guess_pile = L.guess(
        "pile",
        question="Each step of the loop calls `backward(loss)`, which gives every parameter a `grad`. What do you think happens if nothing sets the `grad`s back to 0 between the steps?",
        options={
            "overwrite": "Nothing: each backward pass writes new slopes over the old ones",
            "right": "Each step moves by the slopes of all the steps so far, added up",
            "error": "Python stops with an error at the second step",
        },
        reveal="**They pile up.** `backward_step()` adds to what `grad` already holds, so after ten steps each `grad` holds ten slopes, from ten different places. Each step then moves by all of them together, further and further, and in a direction that is no longer the slope where you are. So the loop sets every `grad` to 0 before each backward pass.",
    )
    guess_pile
    return (guess_pile,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **training loop** repeats the same steps, many times:

        1. **Forward**: the model's prediction for every example, and from them the loss, here the mean squared error.
        2. **Zero the gradients**: set the `grad` of every parameter to 0, because `backward_step()` adds to it.
        3. **Backward**: `backward(loss)` gives every parameter its slope.
        4. **Step**: move every parameter against its slope, by the learning rate $\eta$ times that slope:
        """),
        L.formula(r"[[param:w]] \leftarrow [[param:w]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial w}]] \quad \text{for every parameter } [[param:w]]"),
        mo.md(r"""
        Your `Value` cannot subtract or divide, but it can add and multiply. The residual $y - \hat{y}$ only counts squared, and $(\hat{y} - y)^2$ is the same number, so `mse` works with $\hat{y} - y$: `y_pred + Value(-y)`. The mean is the sum times `Value(1 / n)`. The loss of every step is kept in a list, so you can see it go down.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **training loop** | | the steps that train a model, again and again: forward and loss, zero the gradients, backward, step | `for step in range(steps):` |
        | **zero the gradients** | | setting the `grad` of every parameter to 0 before the backward pass, because the slopes add up | `p.grad = 0` |
        | **step** | | moving every parameter against its slope: `p.data - lr * p.grad` | `p.data = p.data - lr * p.grad` |
        | `mse(model, xs, ys)` | | the mean squared error of a model on the examples `xs` with labels `ys`, as a `Value` | `loss = mse(m, xs, ys)` |
        """),
        L.watch_out("Zero the gradients before `backward(loss)`, not after it. If you zero them after the backward pass and before the step, every slope is 0 when the step comes, and the model never changes."),
        mo.md(r"""
        Move the slider, or drag across the loss chart: it sets the number of steps an `MLP(1, [8, 1])` has trained on 12 points that lie on a curve, with $\eta$ = 0.2. One chart shows the points and <span class="c-pred">what the model predicts</span>; the other, the <span class="c-loss">loss</span> of every step, with a dot at the number of steps you chose.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    def _frame(steps):
        steps = int(steps)
        losses, grid, shots = C.snapshots()
        said = (f"""After <span class="ea-num">{steps}</span> steps the model is untrained: a line from 8 ReLU neurons with random weights. The loss is <span class="c-loss ea-num">{L.number(losses[0], 4)}</span>.""" if steps == 0 else
                f"""After <span class="ea-num">{steps}</span> steps the loss is <span class="c-loss ea-num">{L.number(losses[steps], 4)}</span>. Each ReLU neuron adds a bend to the line; the loop moves the bends to where the points need them.""")
        return (C.picture(steps, {"fit": "The points and the model", "loss": "The loss of each step", "step": "step", "loss_y": "loss",
                                  "alt": f"12 points on a curve that is lowest in the middle, and the line of an MLP after {steps} steps of training. A second chart shows the loss of every step, falling fast at first and then slowly, from {L.number(losses[0], 2)} to {L.number(losses[-1], 4)}" + (f", with a dot at step {steps}" if steps else "") + "."},
                          losses=losses, grid=grid, curve=shots[steps], drag=1),
                said)

    explore_steps = L.explore("steps", "steps", _frame, start=0, low=0, high=300, step=25, role="", task="After how many steps is the loss below 0.01? And how much does it still fall in the last 150 steps?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    After 50 steps the loss is below 0.01: 0.0099, from 0.3539 at the start. In the last 150 steps it only goes from 0.0028 to 0.0018: the bends are almost where the points need them, so the slopes are small, and so is each step.

    ### Recipe

    To train a model with your project:

    1. Forward: the loss of the model on all examples, `loss = mse(model, xs, ys)`.
    2. Zero the gradients: `p.grad = 0` for every `p` in `model.parameters()`.
    3. Backward: `backward(loss)`.
    4. Step: `p.data = p.data - lr * p.grad` for every parameter.
    5. Keep `loss.data`, and go back to step 1, `steps` times.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The same loop without your project: a line $\hat{y} = w \cdot x + b$ on three points, with the slopes of the mean squared error from the chain rule, as in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en), but with `_r` as $\hat{y} - y$, so without the minus sign in front. The slopes are sums over the examples, so they start at 0 in every step. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_loop():
    _xs = [1, 2, 3]
    _ys = [2, 4, 6]
    _w = 0.0
    _b = 0.0
    _lr = 0.1
    _n = len(_xs)
    _losses = []
    for _step in range(6):
        # 1. forward: the mean squared error
        _loss = 0
        for _k in range(_n):
            _r = _w * _xs[_k] + _b - _ys[_k]
            _loss = _loss + _r * _r / _n
        # 2. zero the gradients: the slopes start at 0
        _grad_w = 0
        _grad_b = 0
        # 3. backward: add the slope of every example
        for _k in range(_n):
            _r = _w * _xs[_k] + _b - _ys[_k]
            _grad_w = _grad_w + 2 * _r * _xs[_k] / _n
            _grad_b = _grad_b + 2 * _r / _n
        # 4. step: against the slopes
        _w = _w - _lr * _grad_w
        _b = _b - _lr * _grad_b
        # 5. keep the loss, and again
        _losses.append(_loss)
        print(_step, _loss, _w, _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The loss falls from about 18.67 to about 0.30 in one step, and then slowly: $w$ and $b$ move towards a line through the three points, $w = 2$ and $b = 0$. If `_grad_w = 0` stood before the loop instead of inside it, the slopes of the first step would still be in `_grad_w` at the second, and each step would move by all of them together. In your project, `backward` does step 3 for every parameter of an MLP at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. In which order?
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="Which order of the steps trains a model?",
        options={
            "late": "Forward and loss, backward, zero the gradients, step",
            "first": "Zero the gradients, step, forward and loss, backward",
            "right": "Forward and loss, zero the gradients, backward, step",
        },
        correct="right",
        why={
            "late": "When the step comes, what are the grads? See the watch-out in *The idea*.",
            "first": "The step comes right after the gradients were set to 0: what does it move by? See the watch-out in *The idea*.",
            "right": "The grads are 0 when `backward` starts, so each holds exactly the slope of this step, and the step uses them. Zeroing the gradients may also come first, before the forward pass, as long as it comes before `backward`.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_exercise_mse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loss of a model

        Finish `mse(model, xs, ys)`: the mean squared error of `model` on the examples `xs`, a list with one list of numbers per example, and the labels `ys`. The box imports your `Value`, and the loop already makes the inputs of each example into `Value`s and gives the prediction `y_pred`. Replace the two `...`: in the loop, add the square of the residual to `total`; after the loop, give back the mean, as a `Value`. The check calls it on MLPs from your project, and runs your `backward` on what it gives. When it is right, the check saves it in your project, as `work/projects/nn/loss.py`.
        """),
        L.task("Make `mse` give back the mean squared error of a model, as a Value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mse():
    box_mse = L.code_box("mse", lines=14)
    box_mse
    return (box_mse,)


@app.cell(hide_code=True)
def check_mse(box_mse):
    result_mse = L.check("mse", box_mse.value)
    result_mse
    return (result_mse,)


@app.cell(hide_code=True)
def hints_mse():
    L.hints(
        "mse",
        [
            "Step 1 of the recipe, with your `Value`: for each example $\\hat{y} - y$, its square, the sum, and then the mean.",
            "Your `Value` can only add and multiply: $\\hat{y} - y$ is `y_pred` plus a `Value` of minus the label, a square is a value times itself, and the mean is the sum times a `Value` of one over the number of examples.",
            "`r = y_pred + Value(-ys[k])`, then `total = total + ___ * ___`; after the loop, `return total * Value(___)`.",
        ],
        why="With every parameter of `MLP(1, [3, 1])` at 0.5 and the example 1 with label 2: each hidden neuron gives $0.5 + 0.5 = 1$, the output $0.5 + 3 \\cdot 0.5 = 2$, so the residual is 0 and the loss 0. Each residual adds its square, and the sum times $\\frac{1}{n}$ is the mean.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your training loop

        Finish `train(model, xs, ys, lr, steps)`. It imports your `backward` and your `mse` from your project, works out the loss in every step, and keeps `loss.data` in `losses`. Replace the `...` with steps 2, 3 and 4 of the recipe. The check trains MLPs of different sizes from the same starting weights as the course, and compares the losses of every step. When it is right, the check saves it in your project, as `work/projects/nn/train.py`.
        """),
        L.task("Make `train` zero the gradients, run the backward pass and take a step, in every round of the loop."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=14)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_mse):
    result_mse  # check again when exercise 2 saves your module
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Steps 2, 3 and 4 of the recipe, in that order, inside the loop: the worked example does the same with two parameters.",
            "Steps 2 and 4 are each a loop over `model.parameters()`; step 3 is one call.",
            "`for p in model.parameters():` with `p.grad = 0`; then `backward(___)`; then `for p in model.parameters():` with `p.data = p.data - ___ * ___`.",
        ],
        why="Zero the gradients first, so that `backward(loss)` fills every `grad` with the slope of this step only; then every parameter moves against it. The check compares the loss of every step with the course's own run from the same starting weights.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below does what the picture in *The idea* does, but with your own code: it builds `MLP(1, [8, 1])` from the same starting weights and trains it with your own `train`, 300 steps with $\eta$ = 0.2. The cell uses your own modules, so it works once exercise 3 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_train(result_train):
    result_train  # run again when exercise 3 saves your module
    _fit = C.trained()
    if _fit is None:
        _out = mo.md("*Finish exercise 3 first: then your `train` is in your project, and this cell uses it.*")
    elif not _fit:
        _out = mo.md("*Your `loss.py` or `train.py` in `work/projects/nn/` no longer works the way its check expects. Did you change one of them? Delete it, and check its exercise again: its check then saves it again.*")
    else:
        _losses, _grid, _curve, _course_files = _fit
        _out = mo.vstack([
            mo.Html(C.picture(len(_losses), {"fit": "Your MLP after training", "loss": "Your loss of each step", "step": "step", "loss_y": "loss",
                                             "alt": f"12 points on a curve and the line of the MLP that your train function trained. A second chart shows its loss of every step, from {L.number(_losses[0], 2)} to {L.number(_losses[-1], 4)}."},
                              losses=_losses, grid=_grid, curve=_curve)),
            mo.md(f"""Your loop took the loss from <span class="c-loss ea-num">{L.number(_losses[0], 2)}</span> to <span class="c-loss ea-num">{L.number(_losses[-1], 4)}</span>. {"Every part of it is yours: the `Value` and its slopes, the backward pass, the neurons and layers, the loss and the loop. That is a whole neural network library, small but complete." if not _course_files else "The loss and the loop are yours. For these files, the run used the course's version: " + ", ".join(_course_files) + "."}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en) took steps against the slopes of two parameters; now your training loop does that for every parameter of the MLP of [Neurons, layers and an MLP on top of your Value](/course/open/nn-micrograd-mlp?lang=en), with the slopes of your own `backward`, on the mean squared error of [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en). Your project `nn` now has `loss.py` and `train.py`. In the next lesson, *PyTorch tensors*, you meet PyTorch, which does all of this for whole arrays at once, and much faster.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_mse, result_train):
    L.wrap_up(choice_order, result_mse, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
