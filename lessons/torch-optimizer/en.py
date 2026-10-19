# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-optimizer. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-optimizer", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en) and [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="`net` has the layers `self.hidden = nn.Linear(2, 3)` and `self.out = nn.Linear(3, 1)`. How many tensors does `net.parameters()` give?",
        options={
            "layers": "2: one per layer",
            "numbers": "13: one per number",
            "right": "4: a `weight` and a `bias` for each layer",
        },
        correct="right",
        why={
            "layers": "Each layer has two parameters, its `weight` and its `bias`, as the worked example showed with `for _p in _net.parameters():`. See *All parameters at once* in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en).",
            "numbers": "13 is the number of numbers in them. Each `weight` and each `bias` is one tensor. See [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en).",
            "right": "`hidden.weight` (3, 2), `hidden.bias` (3,), `out.weight` (1, 3) and `out.bias` (1,): four tensors with 13 numbers in all.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_update():
    quick_update = L.choice(
        "quick-update",
        question="An update step moves $w$ against the slope of the loss, by the learning rate times the slope. At the bottom of the valley the slope is 0. What does an update step do there?",
        options={
            "lr": "It moves $w$ by the learning rate",
            "back": "It sends $w$ back up the valley",
            "right": "Nothing: $w$ stays where it is",
        },
        correct="right",
        why={
            "lr": "The step is the learning rate times the slope. What is any number times 0? See the update step in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).",
            "back": "With a slope of 0 there is no direction to move in. See the update step in [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en).",
            "right": "The learning rate times 0 is 0, so $w$ stays put: gradient descent stops by itself at the bottom.",
        },
    )
    quick_update
    return (quick_update,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Each step of your training loop in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en) had two loops of its own: one to set every `.grad` to `None`, and one for the update step of [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en), inside `torch.no_grad()`. Every network needs the same two loops. And the update step is the part people keep improving: there are ways to take smarter steps than "the learning rate times the slope", and each one would change that loop. PyTorch puts the step in an object of its own, an **optimizer**: you give it the parameters once, and from then on two short calls do both loops.

    **An everyday example.** A washing machine has programs: you load it once, press start, and it does the washing, rinsing and spinning. You do not stand next to it to run every part yourself, and a new machine with a better program still has the same start button.
    """)
    return


@app.cell(hide_code=True)
def predict_early():
    guess_early = L.guess(
        "early",
        question="`opt = torch.optim.SGD(net.parameters(), lr=0.1)` is made, and `opt.step()` is called before any `loss.backward()`. What do you think happens?",
        options={
            "error": "Python stops with an error: there are no slopes yet",
            "right": "Nothing: no parameter moves",
            "random": "Every parameter moves a random bit",
        },
        reveal="**Nothing.** Before the first backward pass, every `.grad` is `None`, and `opt.step()` skips every parameter without a `.grad`. It moves only what `loss.backward()` gave a slope. So a loop that forgets `loss.backward()` runs without an error, and the loss just never changes.",
    )
    guess_early
    return (guess_early,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **optimizer** is an object that takes the update step for a list of parameters. You make it once, before the loop, from the parameters it should change and the learning rate: `opt = torch.optim.SGD(net.parameters(), lr=0.1)`. It keeps them, and two of its methods do the work:

        - **`opt.zero_grad()`** sets the `.grad` of every one of its parameters to `None`: the first of your two loops.
        - **`opt.step()`** moves every parameter that has a `.grad` against its slope, by the learning rate times that slope, without recording it: the second loop, with its `torch.no_grad()`.

        `SGD` takes exactly the update step of gradient descent, for every parameter $w$:
        """),
        L.formula(r"[[param:w]] \leftarrow [[param:w]] - [[hyper:\eta]] \cdot [[grad:\frac{\partial \mathcal{L}}{\partial w}]]"),
        mo.md(r"""
        The name SGD comes from *stochastic gradient descent*, which the next lesson explains; for now it is the update step you know. A step of the training loop is then four lines: the forward pass with the loss, `opt.zero_grad()`, `loss.backward()` and `opt.step()`. `opt.zero_grad()` may also come last, after `opt.step()`, as long as the next `loss.backward()` finds no old slopes. One step of your loop, and the same step with an optimizer:
        """),
        C.picture({"own": "Your loop", "opt": "With an optimizer", "1": "every .grad to None: opt.zero_grad()",
                   "2": "the update step, inside torch.no_grad(): opt.step()"}, title="One step of the training loop"),
        mo.md(r"""
        Other optimizers, such as Adam, take smarter steps, with the same two calls; a later lesson explains them. To try one, only the line that makes `opt` changes.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **optimizer** | | an object that takes the update step for the parameters you gave it | `opt = torch.optim.SGD(net.parameters(), lr=0.1)` |
        | `torch.optim.SGD` | s g d | the optimizer that takes the step of gradient descent: every parameter minus the learning rate times its slope | `torch.optim.SGD(params, lr=lr)` |
        | `opt.zero_grad()` | zero grad | sets the `.grad` of every parameter of the optimizer to `None` | `opt.zero_grad()` |
        | `opt.step()` | | one update step for every parameter that has a `.grad` | `opt.step()` |
        """),
        L.watch_out("An optimizer keeps the parameters it was made with. Make a new network, and the old optimizer still works only on the parameters of the old one, so the new network never moves: make a new optimizer too, from the new `net.parameters()`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train with an optimizer:

    1. Make the network, and then the optimizer from its parameters: `opt = torch.optim.SGD(net.parameters(), lr=lr)`.
    2. In each step, work out the loss with a forward pass.
    3. `opt.zero_grad()`, then `loss.backward()`.
    4. `opt.step()`, and keep `loss.item()` if you want to follow the loss.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The XOR network from the worked example in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en), with 4 hidden neurons and seed 1, now trained with an optimizer. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_optimizer():
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.out = nn.Linear(4, 1)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    _X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    # 1. the network, and the optimizer from its parameters
    torch.manual_seed(1)
    _net = _Net()
    _opt = torch.optim.SGD(_net.parameters(), lr=0.1)
    print(_opt)
    _losses = []
    for _step in range(300):
        # 2. the forward pass and the loss
        _loss = ((_net(_X) - _y) ** 2).mean()
        # 3. no old slopes, then the backward pass
        _opt.zero_grad()
        _loss.backward()
        # 4. the step, and the loss as a number
        _opt.step()
        _losses.append(_loss.item())
    print(_losses[0], _losses[50], _losses[100])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `print(_opt)` shows the optimizer with its settings for `Parameter Group 0`, the parameters it got: `lr: 0.1`, and others that stay 0 or off here. The loss goes from about 0.44 at the start to about 0.11 after 50 steps and 0.014 after 100: the same numbers as with your own two loops in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en), because `SGD` takes the same steps. Only the loop got shorter.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stale():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A new network, an old optimizer
    """)
    return


@app.cell(hide_code=True)
def choice_stale_cell():
    choice_stale = L.choice(
        "stale",
        question="`opt = torch.optim.SGD(net.parameters(), lr=0.1)` is made. Later you make a new network, `net = Net()`, and train it with the same `opt`. What happens?",
        options={
            "follows": "The new network trains: `opt` looks up the parameters of `net` at each step",
            "right": "The new network does not change: `opt` holds the parameters of the old one",
            "error": "Python stops with an error at the first `opt.step()`",
        },
        correct="right",
        why={
            "follows": "`opt` got the parameters themselves when it was made, not the name `net`. See the watch-out in *The idea*.",
            "error": "As far as Python knows, nothing is wrong: `opt` has parameters, they just belong to the old network. See the watch-out in *The idea*.",
            "right": "`opt.zero_grad()` and `opt.step()` only touch the parameters `opt` was made with. The new network gets slopes from `loss.backward()`, but nothing moves them. Make a new optimizer from the new `net.parameters()`.",
        },
    )
    choice_stale
    return (choice_stale,)


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loop with an optimizer

        The network of exercise 2 in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en), with 3 hidden neurons and seed 0, and its optimizer are there, with the forward pass in the loop. Replace the `...` with the rest of each step: steps 3 and 4 of the recipe. The last line is the loss of the last step, as a Python number. The check also tries another learning rate.
        """),
        L.task("Let the optimizer zero the gradients and take the step, around the backward pass."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loop():
    box_loop = L.code_box("loop", lines=24)
    box_loop
    return (box_loop,)


@app.cell(hide_code=True)
def check_loop(box_loop):
    result_loop = L.check("loop", box_loop.value)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "loop",
        [
            "Steps 3 and 4 of the recipe, as in the worked example.",
            "Three calls, each on its own line inside the loop: one on `opt` to clear the old slopes, one on `loss`, and one on `opt` for the step.",
            "`opt.zero_grad()`, `loss.___()` and `opt.___()`.",
        ],
        why="`opt.zero_grad()` clears the slopes of the step before, `loss.backward()` gives every parameter its new slope, and `opt.step()` moves each one against it. After 500 steps the network gives almost 0, 1, 1 and 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A function that trains

        Finish `train(net, X, y, lr, steps)`: it makes its own optimizer from the parameters of `net` with the learning rate `lr`, takes `steps` steps, and gives back the loss of every step in a list, as Python numbers. Replace the two `...`. The last line trains the network of exercise 2 for 5 steps. The check also tries a new network with another learning rate and number of steps.
        """),
        L.task("Make `train` train any network with an optimizer, and give back its losses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=26)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Step 1 of the recipe before the loop, and steps 3 and 4 in it, as in the worked example.",
            "The optimizer gets the parameters of the `net` that `train` was given, and `lr`. In the loop, after the three calls, `losses` gets the loss as a Python number.",
            "`opt = torch.optim.SGD(net.___(), lr=___)`; in the loop `opt.zero_grad()`, `loss.backward()`, `opt.step()` and `losses.append(loss.___())`.",
        ],
        why="`train` makes the optimizer from the `net` it gets, so it works for any network. Each step keeps `loss.item()`, the loss before that step: 1.3444 first, then lower each time.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** An optimizer is the update step of [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en), taken for all parameters of a network at once: `opt.zero_grad()` and `opt.step()` replace the two loops of [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en). Your `train` now works for any network. In the next lesson, *Mini-batches and epochs*, each step learns from a small part of the data instead of all of it, and you will see where the S of SGD comes from.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stale, result_loop, result_train):
    L.wrap_up(choice_stale, result_loop, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
