# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-activation-stats. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-activation-stats", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading loss curves](/course/open/torch-loss-curves?lang=en) and [Initialization](/course/open/nn-initialization?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_guessing():
    quick_guessing = L.choice(
        "quick-guessing",
        question="The training loss and the validation loss of a classifier for the ten digits both stay at about 2.30. What is 2.30 here?",
        options={
            "right": "The loss of guessing, with a probability of 0.1 for each digit",
            "accuracy": "The number of digits out of 10 that it gets right",
            "overfitting": "The loss at which overfitting starts, for any model of ten digits",
        },
        correct="right",
        why={
            "accuracy": "A cross-entropy loss is not a count of right answers. 2.30 is $-\\log 0.1$: the loss when the right digit gets a probability of 0.1. See *Stuck* in [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "overfitting": "Overfitting shows as a validation loss that rises while the training loss falls; here both stay put. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "$-\\log 0.1 \\approx 2.30$: the model gives every digit the same probability. The curves are stuck, and they do not say why.",
        },
    )
    quick_guessing
    return (quick_guessing,)


@app.cell(hide_code=True)
def quick_check_he():
    quick_he = L.choice(
        "quick-he",
        question="He initialization doubles the variance of the starting weights for ReLU. Why?",
        options={
            "right": "ReLU sets about half of the z's to 0, which halves the signal",
            "pieces": "ReLU is made of two straight pieces, so each needs its own share of the spread",
            "slope": "The slope of ReLU is 2 above 0, so the signal doubles in every layer",
        },
        correct="right",
        why={
            "pieces": "The two pieces do not each get a spread. What matters is that ReLU gives 0 for about half of its inputs. See [Initialization](/course/open/nn-initialization?lang=en).",
            "slope": "The slope of ReLU is 1 above 0, not 2. See [Initialization](/course/open/nn-initialization?lang=en).",
            "right": "About half of the $z$'s are below 0 and come out as 0. Today you look at neurons that give 0 for every example.",
        },
    )
    quick_he
    return (quick_he,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The loss is one number for the whole network. When it is stuck, as in [Reading loss curves](/course/open/torch-loss-curves?lang=en), it tells you *that* something is wrong, not *where*. A network with three hidden layers has three places where it can go wrong, and each layer can show it: neurons that never give anything but 0, or numbers that grow layer after layer. To see that, you look inside: at what each layer gives for many examples at once, and at the size of its gradients. PyTorch lets you do that without changing the model.
    """)
    return


@app.cell(hide_code=True)
def predict_dead():
    guess_dead = L.guess(
        "how-many-dead",
        question="An MLP with three hidden layers of 128 ReLU neurons trains for one epoch on the 6,000 training digits, with a learning rate of 2.0. Its loss ends at 2.30: it guesses. How many of the 128 neurons of the third hidden layer do you think give 0 for every one of the 1,000 test digits?",
        options={
            "tenth": "About 13: a tenth of them",
            "half": "About 64: ReLU switches off about half",
            "right": "All 128: the whole layer switches off",
        },
        reveal="**All 128.** Every $z$ of the third layer is 0 or below, for every test digit, so after ReLU the layer gives only zeros. The output layer then gets the same input for every digit, and can only guess. And it stays that way: below 0 the slope of ReLU is 0, so no gradient comes back through these neurons, and their weights never change again.",
    )
    guess_dead
    return (guess_dead,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A ReLU neuron whose $z$ is 0 or below for every example is a **dead neuron**: it always gives 0. Below 0, the slope of ReLU is 0, so every gradient through the neuron is 0 too, for its weights and its bias term:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_{jk}}]] = \sum_i [[grad:\frac{\partial \mathcal{L}}{\partial h_{ij}}]] \cdot \mathrm{ReLU}'(z_{ij}) \cdot x_{ik} = 0 \qquad \text{when } z_{ij} \le 0 \text{ for every example } i"),
        mo.md(r"""
        Here $j$ is the neuron, $k$ one of its inputs and $i$ the example. No step moves these weights. Only a change in the layers before can still bring such a neuron back; in the first layer, nothing can. One step that is far too large can push the bias terms and weights of many neurons that far down, and a whole layer can die. When a layer gives only zeros, nothing comes through it, and the layers before it get no gradient either.

        To find dead neurons, you look at what each layer gives. A **forward hook** is a function that PyTorch calls every time a layer has computed its output. `layer.register_forward_hook(keep)` hangs the function `keep` on `layer`; it gets three things, the layer, its inputs and its output, and can keep the output in a list. Hooks run in the order the model runs its layers. `register_forward_hook` gives back a handle, and `handle.remove()` takes the hook off again. A hook on an `nn.Linear` gets its $z$'s, one row per example and one column per neuron. A neuron is dead when its whole column is 0 or below: `(Z <= 0).all(dim=0)` gives `True` for each such column.

        An **activation histogram** is a histogram of what one layer gives, for many examples at once. Here it shows the $z$'s, before ReLU: after ReLU, everything below 0 would pile up at 0, and you could not see how far below it lies. For the $z$'s of a healthy ReLU layer, it lies on both sides of 0; for a dying layer, it slides below 0. And $z$'s that get larger layer after layer, or gradients that are much larger in one layer than in the others, point to exploding gradients, as in [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **dead neuron** | | a ReLU neuron whose $z$ is 0 or below for every example: it always gives 0, and gets no gradient | `(Z <= 0).all(dim=0)` |
        | **forward hook** | | a function that PyTorch calls every time a layer has computed its output | `layer.register_forward_hook(keep)` |
        | `handle.remove()` | handle remove | takes a hook off its layer again | `handle = layer.register_forward_hook(keep)` |
        | **activation histogram** | | a histogram of what a layer gives, for many examples at once | |
        | `.all(dim=0)` | all, dim zero | for each column: `True` when every value in it is `True` | `(Z <= 0).all(dim=0)` |
        """),
        L.watch_out("Take the hooks off when you are done. A hook stays on its layer until you remove it: every later run of the model calls it again, also during training, and a list that it fills keeps growing. And run the model under `torch.no_grad()` while you look, so that the kept outputs do not hold on to a whole graph."),
        mo.md(r"""
        The picture: activation histograms of the third hidden layer, its $z$'s for all 128 neurons and the 1,000 test digits, after one epoch with $\eta = 0.3$ and with $\eta = 2.0$. With 0.3, the $z$'s lie on both sides of 0, most between −1 and 2, and 2 of the 128 neurons are dead. With 2.0, every $z$ is 0 or below, most of them just below: ReLU turns all of them into 0.
        """),
        mo.Html(C.z_picture({"good": "η = 0.3: {dead} of 128 neurons dead", "large": "η = 2.0: {dead} of 128 neurons dead",
                             "x": "z in hidden layer 3", "y": "share of the z's", "zero": "← ReLU gives 0",
                             "alt": "Two histograms of the z's of hidden layer 3. With a learning rate of 0.3, they lie between about −1 and 2 on both sides of 0, highest just below 0. With 2.0, every z is 0 or below: 70% lies just below 0, the rest down to about −1."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look inside a network:

    1. Choose the layers to look at, and a fixed batch of examples, such as the test digits.
    2. Hang a forward hook on each layer that keeps its output.
    3. Run the model once on the batch, under `torch.no_grad()`, and take the hooks off again.
    4. Per layer: the share of dead neurons, and a histogram of its $z$'s.
    5. For the gradients: one backward pass on the batch, then the mean size of the gradient of each layer's weights. A layer with gradients of 0, or much smaller or larger than the others, is where it goes wrong.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The network of the guess, trained for one epoch with a learning rate of 2.0; the first cell trains it, in a few seconds at most. The second cell looks inside; in step 5, `model.zero_grad()` does what `opt.zero_grad()` does, for every weight and bias term of the model, so that the gradients of the last training step do not count, and `.abs()` and `.mean()` work on a tensor as `np.abs` and `np.mean` on an array. The comments with `#` say which step of the recipe each part is. Step 5 is not finished: run the cells as they are, then complete step 5 where the comment says, and run it again. After that, change `lr=2.0` in the first cell into `lr=0.3`: the second cell then runs again by itself.
    """)
    return


@app.cell
def example_train():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.h1 = nn.Linear(784, 128)
            self.h2 = nn.Linear(128, 128)
            self.h3 = nn.Linear(128, 128)
            self.out = nn.Linear(128, 10)

        def forward(self, x):
            return self.out(torch.relu(self.h3(torch.relu(self.h2(torch.relu(self.h1(x)))))))

    _data = np.load("data/qmnist-small.npz")
    _X = torch.tensor(_data["train_images"], dtype=torch.float32) / 255
    _y = torch.tensor(_data["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(_data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(_data["test_labels"], dtype=torch.int64)
    # one epoch of training with SGD
    torch.manual_seed(0)
    model = DeepMLP()
    _loader = DataLoader(TensorDataset(_X, _y), batch_size=64, shuffle=True)
    _opt = torch.optim.SGD(model.parameters(), lr=2.0)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(2)
    for _xb, _yb in _loader:
        _loss = loss_fn(model(_xb), _yb)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    print(_loss.item())
    return X_test, loss_fn, model, torch, y_test


@app.cell
def example_look(X_test, loss_fn, model, torch, y_test):
    # 1. the three hidden layers, and the 1,000 test digits
    _layers = [model.h1, model.h2, model.h3]
    # 2. on each layer a forward hook that keeps its output, the z's
    _zs = []

    def _keep(module, inputs, output):
        _zs.append(output)

    _handles = []
    for _layer in _layers:
        _handles.append(_layer.register_forward_hook(_keep))
    # 3. one run without gradients, then the hooks off again
    with torch.no_grad():
        model(X_test)
    for _handle in _handles:
        _handle.remove()
    # 4. per layer: the share of neurons whose z is 0 or below for every test digit
    for _z in _zs:
        print((_z <= 0).all(dim=0).float().mean().item())
    # 5. one backward pass on the test digits
    _test_loss = loss_fn(model(X_test), y_test)
    model.zero_grad()
    _test_loss.backward()
    print(_test_loss.item())
    # then: for each layer in _layers, print the mean size of the gradient of its weights,
    # _layer.weight.grad.abs().mean().item(), and run again
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With a learning rate of 2.0, the last mini-batch has a loss of 2.29, and the test digits 2.30: the loss of guessing. Inside, more than half of the neurons of the first hidden layer are dead, most of the second, and all of the third. The course measured 59% and 80%; your computer may print a little more or less for the first two, because at a learning rate this large the smallest rounding difference grows. Whatever digit comes in, the third layer gives only zeros. Step 5 shows what that means for training: the gradient of the weights of every hidden layer is exactly 0. No step moves them any more, and more epochs will not help. Only the bias terms of the output layer still learn, and they can only learn how often each digit comes.

    With 0.3, the last mini-batch has a loss of 0.79, and the test digits 0.83. No neuron of the first layer is dead, 1 of the second and 2 of the third (the cell prints their shares, `0.0078125` and `0.015625`), and the gradients are about 0.001 to 0.002 in every layer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dead():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A dead neuron goes on training
    """)
    return


@app.cell(hide_code=True)
def choice_dead_cell():
    choice_dead = L.choice(
        "what-dead-means",
        question="A neuron in the first hidden layer of an MLP gives 0 for every training example. You go on training with SGD. What happens to its weights?",
        options={
            "grow": "They grow: the loss keeps pushing the neuron back up above 0, step by step",
            "right": "Nothing: every gradient through it is 0, so no step of SGD moves them",
            "shrink": "They shrink slowly to 0, like weights that the model does not need",
        },
        correct="right",
        why={
            "grow": "The loss can push only through the slope of ReLU, and below 0 that slope is 0: no gradient reaches the weights of this neuron.",
            "shrink": "Shrinking takes steps too, and every step of SGD is the learning rate times the gradient: here 0.",
            "right": "Below 0 the slope of ReLU is 0, so for every example the gradient of the neuron's weights and bias term is 0. In the first layer, its inputs, the pixels, never change either: it stays dead for good. In a later layer, a change in the layers before can still bring a neuron back.",
        },
    )
    choice_dead
    return (choice_dead,)


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The share of dead neurons

        Write `dead_share(Z)`: `Z` holds the $z$'s of one layer, one row per example and one column per neuron, as a hook gives them. Give back the share of dead neurons, as a number: the share of the columns in which every $z$ is 0 or below. The check calls `dead_share(torch.tensor([[1.0, -2.0, 0.0], [-1.0, -0.5, 2.0]]))`, which should give `0.3333`, because only the middle neuron is dead, and then tries other layers.
        """),
        L.task("Make `dead_share` give the share of the neurons whose z is 0 or below for every example."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("dead-share", lines=5)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("dead-share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "dead-share",
        [
            "`Z <= 0` gives a tensor of `True` and `False` with the shape of `Z`. `.all(dim=0)` asks, for each column, whether everything in it is `True`.",
            "`.float()` turns `True` into 1.0 and `False` into 0.0, and `.mean()` then gives the share of `True`, as with the accuracy in [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en). `.item()` makes a plain number of it.",
        ],
        why="`return (Z <= 0).all(dim=0).float().mean().item()`. In the example, the columns are $[1, -1]$, $[-2, -0.5]$ and $[0, 2]$; only the middle one is 0 or below everywhere, so 1 of the 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_layers():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Every layer at once, with hooks

        Write `dead_per_layer(model, layers, x)`: `layers` is a list of layers of `model`, in any order. Hang a forward hook on each, run the model once on `x`, take the hooks off again, and give back a list with the share of dead neurons of each layer, in the order of `layers`, by steps 2 to 4 of the recipe. The check makes a small network `model` like the one of the worked example, with two layers, `model.h1 = nn.Linear(4, 6)` and `model.h2 = nn.Linear(6, 5)`, and ReLU between them, and 4 random examples, and calls `dead_per_layer(model, [model.h1, model.h2], x)`, which should give `[0.1667, 0.0]`. Then it gives the layers in the other order, `[model.h2, model.h1]`, and only `[model.h2]`.
        """),
        L.task("Make `dead_per_layer` give the share of dead neurons of each layer, measured with forward hooks, in the order of `layers`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layers():
    box_layers = L.code_box("dead-per-layer", lines=18)
    box_layers
    return (box_layers,)


@app.cell(hide_code=True)
def check_layers(box_layers):
    result_layers = L.check("dead-per-layer", box_layers.value)
    result_layers
    return (result_layers,)


@app.cell(hide_code=True)
def hints_layers():
    L.hints(
        "dead-per-layer",
        [
            "The hooks run in the order the model runs its layers, not in the order of `layers`. So keep each output under its own layer, in a dict: in the hook, `zs[module] = output`. A layer can be a key of a dict.",
            "After the run, go through `layers` in their order and work out the share of each from `zs[layer]`, as in exercise 2. Keep the handles in a list, and call `.remove()` on each after the run.",
        ],
        why="`zs = {}` with `def keep(module, inputs, output): zs[module] = output`; for each layer `handles.append(layer.register_forward_hook(keep))`; then `with torch.no_grad(): model(x)`, `.remove()` on each handle, and for each layer in `layers` the share `(zs[layer] <= 0).all(dim=0).float().mean().item()` into a list.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Below 0, ReLU from [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) gives 0 and has a slope of 0. In [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en), a saturated sigmoid let almost no gradient through; a dead neuron lets none through at all, for every example. [Initialization](/course/open/nn-initialization?lang=en) gives every layer a healthy start, but a learning rate that is far too large, as in [Reading loss curves](/course/open/torch-loss-curves?lang=en), can still push neurons below 0 for good. The loss told you *that* the network was stuck; the hooks told you *where*. In the next lesson, *Batch normalization*, a layer standardizes the $z$'s of every neuron over the mini-batch, at every step of training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dead, result_share, result_layers):
    L.wrap_up(choice_dead, result_share, result_layers)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
