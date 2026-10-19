# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-batches. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-batches", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Let the optimizer take the step](/course/open/torch-optimizer?lang=en) and [Generators: producing values one at a time](/course/open/py-generators?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_zero():
    quick_zero = L.choice(
        "quick-zero",
        question="Right after `opt.zero_grad()`, what is the `.grad` of every parameter of `opt`?",
        options={
            "right": "`None`",
            "zeros": "A tensor of zeros, in the shape of the parameter",
            "last": "The slope of the step before",
        },
        correct="right",
        why={
            "zeros": "That is how it once was, and the name suggests it, but now `zero_grad()` throws the slopes away altogether. See *The idea* in [Let the optimizer take the step](/course/open/torch-optimizer?lang=en).",
            "last": "That is what `zero_grad()` clears away, so that the next `loss.backward()` starts without it. See [Let the optimizer take the step](/course/open/torch-optimizer?lang=en).",
            "right": "`opt.zero_grad()` sets the `.grad` of every parameter it holds to `None`: no slope yet, until the next `loss.backward()`.",
        },
    )
    quick_zero
    return (quick_zero,)


@app.cell(hide_code=True)
def quick_check_pairs():
    quick_pairs = L.choice(
        "quick-pairs",
        question="What does this program show?\n\n```python\ndef pairs(items, starts):\n    for s in starts:\n        yield items[s:s + 2]\n\n\nprint(list(pairs([10, 20, 30, 40, 50, 60], [0, 3])))\n```",
        options={
            "first": "`[[10, 20]]`",
            "stop": "`[[10, 20, 30], [40, 50, 60]]`",
            "right": "`[[10, 20], [40, 50]]`",
        },
        correct="right",
        why={
            "first": "`yield` hands out one pair and pauses; `list` asks for the next one, and the loop goes on with the next `s`. See [Generators: producing values one at a time](/course/open/py-generators?lang=en).",
            "stop": "A slice stops just before its second number: `items[0:2]` is `[10, 20]`, two items.",
            "right": "The generator hands out `items[0:2]` and then `items[3:5]`, one at a time, and `list` collects them.",
        },
    )
    quick_pairs
    return (quick_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your `train` from [Let the optimizer take the step](/course/open/torch-optimizer?lang=en) works out the loss over all examples before every step. For the 4 corners of XOR that costs nothing. QMNIST, a collection of handwritten digits, has 60,000 images to train on: each step would first send all of them through the network, and a network needs thousands of steps. Real training does something else. Each step looks at a small part of the data, a **mini-batch**, so one pass over the data gives the model many steps. Each step is a little less precise, but there are far more of them.

    **An everyday example.** To find out whether a big pot of soup needs salt, you taste one spoonful, not the whole pot. One spoon can be a little off, so you stir, add a pinch and taste again. That is much quicker than eating the whole pot every time.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="60,000 training images, mini-batches of 100 images, and every image used once. How many update steps do you think that is?",
        options={
            "one": "1: one pass over the data",
            "right": "600",
            "hundred": "100: one per image in a batch",
        },
        reveal="**600.** 60,000 / 100 = 600 mini-batches, each with its own update step. One pass over all the data, with every example once, is called an **epoch**. Here one epoch is 600 steps, where the loop of the lesson before took 1.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **mini-batch** is a small group of examples, such as 32 or 100, for one update step: the loss of that group, its slopes, and one step. The **batch size** is how many examples go into a mini-batch. One update step on one mini-batch is an **iteration**, and one pass over all the data, with every example once, an **epoch**. With $n$ examples and batch size $B$, an epoch has $n / B$ iterations, rounded up: the last mini-batch may be smaller.

        The loss of a mini-batch is the mean over its own examples only, $B$ of them in a full mini-batch:
        """),
        L.formula(r"[[loss:\mathcal{L}_{\text{batch}}]] = \frac{1}{B} \sum_{i=1}^{B} \left(y_i - [[pred:\hat{y}_i]]\right)^2"),
        mo.md(r"""
        Here $i$ counts the examples of the mini-batch. Its slopes are an estimate of the slopes over all the data: close, but not the same, and a little different for every batch. Gradient descent with these steps is called **stochastic gradient descent**, SGD: *stochastic* means that chance plays a part, here in which examples share a batch. That is where `torch.optim.SGD` gets its name. The optimizer takes the same step as before; what makes the descent stochastic is the batches you feed it. For that, the examples should be in a random order. The data here already is; the next lesson shuffles it again for every epoch.

        **The batches themselves** come from a generator, as in [Generators: producing values one at a time](/course/open/py-generators?lang=en), that hands out one slice of the data at a time. `range` with a third number counts in steps of that size: `range(0, 10, 4)` gives 0, 4 and 8. A slice that runs past the end simply stops at the end, so the last batch is smaller, and no example is left out:

        ```python
        def batches(X, y, size):
            for start in range(0, len(X), size):
                yield X[start:start + size], y[start:start + size]
        ```

        Training then has two loops: one over the epochs, and inside it one over the mini-batches, with a step for each.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **mini-batch** | | a small group of examples for one update step | `xb, yb` |
        | **batch size** | | how many examples go into a mini-batch | `size = 20` |
        | $B$ | B | the batch size in a formula | |
        | **iteration** | | one update step on one mini-batch | |
        | **epoch** | | one pass over all the data, with every example once: $n / B$ iterations, rounded up | `for epoch in range(3):` |
        | **SGD** | s g d | stochastic gradient descent: gradient descent with the slopes of mini-batches | `torch.optim.SGD` |
        | `range(start, stop, step)` | | counts from `start` to just below `stop`, in steps of `step` | `range(0, 10, 4)` |
        """),
        L.watch_out("A smaller batch is not always better. With batches of 2, every step follows the slope of two examples, which can point far from the slope over all the data: the loss keeps jumping and does not settle. Batch sizes such as 32 to 256 are common: many steps per epoch, and each step still close to the real slope."),
        mo.md(r"""
        The picture shows one run: a straight line fitted to 200 examples, $y = 2x + 1$ plus a random scatter, with the same starting weights, learning rate 0.1 and three batch sizes.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_batches():
    mo.vstack([
        mo.Html(C.picture({"epochs": "Loss over all 200 examples", "epoch": "epoch", "loss": "loss", "steps": "Loss of each mini-batch of 20",
                           "step": "iteration",
                           "alt": "Left: the loss over all 200 examples after every step, for 5 epochs, from 0 to 2. All three batch sizes start at 4.9, above the top. With the full batch of 200 it falls slowly, one step per epoch, to 0.69; with mini-batches of 20 it is at 0.32 after one epoch and 0.28 after two; with mini-batches of 2 it drops at once but keeps jumping between about 0.28 and 0.52. Right: the loss of each of the first 30 mini-batches of 20, jumping up and down while it falls."})),
        mo.md(r"""
        Left, each line follows the loss over all 200 examples after every step, and $B$ is the batch size. All three start at 4.9, above the top of the picture. The full batch of 200 takes one step per epoch, a dot on the dashed line, and is still at 0.69 after 5 epochs. Mini-batches of 20 take 10 steps per epoch and are at 0.28 after two. Mini-batches of 2 take 100 steps per epoch, but each step follows just two examples: the fitted line keeps moving around, and the loss keeps jumping between about 0.28 and 0.52. Right, the loss of each mini-batch of 20 jumps up and down, because each batch holds other examples, while the trend goes down.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train with mini-batches:

    1. Write a generator `batches(X, y, size)` that hands out the examples `size` at a time, the last batch maybe smaller.
    2. Make the model and the optimizer, as before.
    3. Loop over the epochs, and inside that over `batches(X, y, size)`.
    4. For each mini-batch: its loss, `opt.zero_grad()`, `loss.backward()` and `opt.step()`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The data of the picture, with mini-batches of 20 for 2 epochs. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_batches():
    import torch
    from torch import nn

    _X, _y = C.data()   # 200 examples, shapes (200, 1) and (200, 1)

    # 1. the generator
    def _batches(X, y, size):
        for start in range(0, len(X), size):
            yield X[start:start + size], y[start:start + size]

    # 2. the model and the optimizer
    torch.manual_seed(1)
    _model = nn.Linear(1, 1)
    _opt = torch.optim.SGD(_model.parameters(), lr=0.1)
    # 3. epochs, and mini-batches in each
    for _epoch in range(2):
        _steps = 0
        for _xb, _yb in _batches(_X, _y, 20):
            # 4. one step for this mini-batch
            _loss = ((_model(_xb) - _yb) ** 2).mean()
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            _steps = _steps + 1
        print(_epoch, _steps, ((_model(_X) - _y) ** 2).mean().item())
    print(_model.weight.item(), _model.bias.item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each epoch takes 10 steps: 200 examples in mini-batches of 20. After the first epoch the loss over all data is about 0.32, after the second about 0.28. The line found is close to $y = 2x + 1$: a weight near 2 and a bias term near 1. The random scatter in the data keeps the loss from reaching 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_last():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The last mini-batch
    """)
    return


@app.cell(hide_code=True)
def choice_last_cell():
    choice_last = L.choice(
        "last-batch",
        question="1,000 examples in mini-batches of 32, each example once. How many iterations does one epoch have, and how big is the last mini-batch?",
        options={
            "floor": "31 iterations, all of 32 examples; the last 8 examples are left out",
            "right": "32 iterations; the last mini-batch has 8 examples",
            "even": "32 iterations of 31.25 examples each",
        },
        correct="right",
        why={
            "floor": "A slice past the end simply stops there, so the last examples form a smaller batch of their own. See *The batches themselves* in *The idea*.",
            "even": "A mini-batch holds whole examples. What happens to the ones that do not fill a full batch? See *The batches themselves* in *The idea*.",
            "right": "31 · 32 = 992, so 31 full mini-batches leave 8 examples, which form the 32nd. Every example is used once.",
        },
    )
    choice_last
    return (choice_last,)


@app.cell(hide_code=True)
def text_exercise_batches():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A generator for mini-batches

        Finish `batches(X, y, size)`: a generator that hands out the examples in order, `size` at a time, each time a slice of `X` with the same slice of `y`. Replace the `...`. The last lines try it on 10 examples with batches of 4, and keep the length of each batch and its first example and label. The check also tries batches of 3.
        """),
        L.task("Make `batches` hand out the data in mini-batches of `size`, the last one maybe smaller."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batches():
    box_batches = L.code_box("batches", lines=15)
    box_batches
    return (box_batches,)


@app.cell(hide_code=True)
def check_batches(box_batches):
    result_batches = L.check("batches", box_batches.value)
    result_batches
    return (result_batches,)


@app.cell(hide_code=True)
def hints_batches():
    L.hints(
        "batches",
        [
            "Step 1 of the recipe. Each batch is a slice of `X` and the same slice of `y`, handed out one at a time.",
            "A loop over the start of each batch: 0, then `size`, then twice `size`, up to `len(X)`. Each round hands out two slices with `yield`.",
            "`for start in range(0, len(X), ___):` and `yield X[start:start + size], y[___]`.",
        ],
        why="With 10 examples and `size` 4, `start` is 0, 4 and 8. The slices are rows 0 to 3, 4 to 7 and 8 to 9: the last one runs past the end and stops there.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_epochs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Three epochs of mini-batches

        The data of the picture, a model, an optimizer and the two loops are there. Replace the `...` with step 4 of the recipe: one update step for each mini-batch. The last line is the loss over all 200 examples after 3 epochs, as a Python number. The check also tries another learning rate.
        """),
        L.task("Give each mini-batch its own update step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_epochs():
    box_epochs = L.code_box("epochs", lines=22)
    box_epochs
    return (box_epochs,)


@app.cell(hide_code=True)
def check_epochs(box_epochs):
    result_epochs = L.check("epochs", box_epochs.value)
    result_epochs
    return (result_epochs,)


@app.cell(hide_code=True)
def hints_epochs():
    L.hints(
        "epochs",
        [
            "Step 4 of the recipe, as in the worked example.",
            "The loss uses the mini-batch, `xb` and `yb`, not all of `X` and `y`. Then the three calls of the optimizer.",
            "`loss = ((model(___) - ___) ** 2).mean()`, then `opt.zero_grad()`, `loss.backward()` and `opt.step()`.",
        ],
        why="3 epochs of 10 mini-batches give 30 steps, each with the slopes of its own 20 examples. The loss over all data ends at about 0.2821, as in the picture.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The training loop of [Let the optimizer take the step](/course/open/torch-optimizer?lang=en) now runs per mini-batch: a generator as in [Generators: producing values one at a time](/course/open/py-generators?lang=en) hands out the batches, and an epoch is one pass over all of them. That turns the gradient descent of the optimizer into stochastic gradient descent. In the next lesson, *Datasets and DataLoaders*, PyTorch hands out the mini-batches for you, and shuffles the examples for every epoch.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_last, result_batches, result_epochs):
    L.wrap_up(choice_last, result_batches, result_epochs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
