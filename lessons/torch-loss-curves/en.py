# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-loss-curves. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-loss-curves", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Log scales and power laws](/course/open/math-log-scales?lang=en) and [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="A loss falls from 1 to 0.1 in the first 5 epochs, and from 0.1 to 0.01 in the next 5. How do the two drops look on a chart with a log scale on the y-axis?",
        options={
            "smaller": "The second drop is ten times shorter",
            "right": "Both drops are equally long",
            "larger": "The second drop is ten times longer",
        },
        correct="right",
        why={
            "smaller": "That is how they look on a normal axis. On a log scale, equal factors are equal steps. See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "larger": "On a log scale, equal factors are equal steps, and both drops are a factor of 10. See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "right": "Both drops divide the loss by 10, and on a log scale every factor of 10 is the same distance. So a log scale still shows a loss that keeps falling close to 0.",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def quick_check_train():
    quick_train = L.choice(
        "quick-train",
        question="The loss of a model on its training examples keeps going down. What does that tell you about its loss on new examples?",
        options={
            "down": "That it goes down too",
            "up": "That it goes up",
            "right": "Nothing yet: only examples it never trained on can tell you",
        },
        correct="right",
        why={
            "down": "A model can also learn its training examples by heart. See [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "up": "Not necessarily: the training loss alone cannot tell. See [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "right": "The training loss says how well the model fits the examples it learns from. How well it does on new ones shows only on examples it never trained on.",
        },
    )
    quick_train
    return (quick_train,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the demo of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), the loss of your MLP went down every epoch. That was the loss on the digits it trains on. To know how long to train, and whether to change something, you need a second loss next to it, one on digits the model never trains on. Two curves together tell you more than either alone: whether the model is still learning, is learning its examples by heart, or is stuck.

    **An everyday example.** Someone who practises with the same 20 exam questions gets more of them right every day. Whether they would pass the exam shows only on questions they have never seen.
    """)
    return


@app.cell(hide_code=True)
def predict_val():
    guess_val = L.guess(
        "val-loss",
        question="An MLP trains for 30 epochs on 1,000 handwritten digits, with a learning rate of 0.5. Its loss on those 1,000 digits falls from 0.98 after the first epoch to 0.0055. What do you think its loss on 500 other digits does?",
        options={
            "along": "It falls along, to close to 0",
            "right": "It falls for a few epochs, then slowly rises",
            "flat": "It stays where it started",
        },
        reveal="**It falls for a few epochs, then slowly rises.** In the worked example below, the loss on the 500 other digits is lowest after epoch 8, at 0.41, and rises to 0.48 by epoch 30, while the loss on the training digits keeps falling. After epoch 8, the model learns its 1,000 digits by heart: it gets better at them, and a little worse at new ones. That is the overfitting of [Overfitting and underfitting](/course/open/ml-overfitting?lang=en), now epoch by epoch.",
    )
    guess_val
    return (guess_val,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        After every epoch, measure two losses, in eval mode and without a graph: the **training loss**, on all the examples the model trains on, and the **validation loss**, on a validation set as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), examples it never trains on. Draw them as two curves in one chart: the training loss solid, the validation loss dashed. The losses soon come close to 0, so a log scale on the y-axis, as in [Log scales and power laws](/course/open/math-log-scales?lang=en), shows what happens there.

        Three patterns tell you what is going on:

        - **Still learning.** Both curves are still going down, close to each other. The model has not learned what it can yet: train for more epochs.
        - **Overfitting.** The training loss keeps going down, but the validation loss reaches a lowest point and rises again. From there on, the model learns its examples by heart. The best model is the one of the epoch with the lowest validation loss; more training examples make learning by heart harder.
        - **Stuck.** Both curves stay high and almost flat. For ten digits, a loss of about 2.30 is the loss of guessing, a probability of 0.1 for each digit. The steps change the model too little: the learning rate is too small, or something in the update step is wrong, such as a missing `opt.step()`.

        A learning rate that is too large, as in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en), gives a fourth pattern: a loss that jumps up and down, or grows. A learning rate that is far too large can also leave both curves stuck at the loss of guessing: stuck curves can come from a step that is too small or one that is far too large.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **training loss** | | the loss on the examples the model trains on, here measured after each epoch | `train_losses` |
        | **validation loss** | | the loss on the validation set, examples the model never trains on | `val_losses` |
        | **train and validation curves** | | the two losses per epoch in one chart: training solid, validation dashed | `plt.plot(train_losses)` |
        | `"--"` | | in `plt.plot`, a dashed line | `plt.plot(val_losses, "--")` |
        """),
        L.watch_out("Both curves jump a little from epoch to epoch: each epoch's mini-batches push the model this way or that, and a few hundred validation examples are a small sample. One epoch up is not yet overfitting: look at the trend over several epochs. And the curves are one run: with another seed, the lowest point can fall several epochs earlier or later: in ten seeds of the middle run below, anywhere from epoch 5 to 13."),
        mo.md(r"""
        The picture: three runs of the MLP with 64 hidden neurons, each on the same 1,000 training digits and 500 validation digits, for 30 epochs, with three learning rates: the training loss solid, the validation loss dashed. Each is one run.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_runs():
    _epoch, _loss = C.best("overfitting")
    mo.vstack([
        mo.Html(C.picture({"learning": "η = 0.05: still learning", "overfitting": "η = 0.5: overfitting", "stuck": "η = 0.001: stuck",
                           "epoch": "epoch", "loss": "loss", "lowest": "lowest",
                           "alt": f"Three charts of the loss per epoch on a log scale, the training loss solid and the validation loss dashed. Left, with a learning rate of 0.05, both go down from about 2 to below 0.5, close together. In the middle, with 0.5, the training loss falls to below 0.01, while the validation loss is lowest after epoch {_epoch}, at {L.number(_loss, 2)}, and then rises. Right, with 0.001, both fall only from 2.29 to 2.16."})),
        mo.md(f"""
        Left, both curves are still going down after 30 epochs. In the middle, the validation loss is lowest after epoch {_epoch}, at <span class="c-loss ea-num">{L.number(_loss, 2)}</span>, while the training loss goes on down to below 0.01. Right, both fall only from 2.29 to 2.16, hardly away from 2.30, the loss of guessing.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read how a training goes:

    1. Keep a validation set apart: examples the model never trains on.
    2. After each epoch, in eval mode and without a graph, measure the loss on all training examples and on the validation set, and append each to its own list.
    3. Draw both in one chart, `plt.plot(train_losses)` and `plt.plot(val_losses, "--")`; with losses close to 0, add `plt.yscale("log")`.
    4. Read the curves: both still falling is still learning; training falling and validation rising is overfitting, and the best epoch is the one with the lowest validation loss; both high and flat is stuck.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The middle run of the picture: an MLP with the same layers as your `DigitsMLP`, trained on the first 1,000 training digits with $\eta$ = 0.5, for 30 epochs of mini-batches of 50. The validation set is 500 other training digits, numbers 5000 to 5499. The cell draws the two curves and prints the best epoch. The comments with `#` say which step of the recipe each part is. Run the cell; it takes a second.
    """)
    return


@app.cell
def example_curves():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt

    class Net(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    # 1. a validation set apart: digits the model never trains on
    X_train, y_train = X[:1000], y[:1000]
    X_val, y_val = X[5000:5500], y[5000:5500]

    torch.manual_seed(0)
    model = Net(64)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)
    torch.manual_seed(1)
    train_losses = []
    val_losses = []
    for epoch in range(30):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 2. after each epoch: both losses, in eval mode and without a graph
        model.eval()
        with torch.no_grad():
            train_losses.append(loss_fn(model(X_train), y_train).item())
            val_losses.append(loss_fn(model(X_val), y_val).item())
        model.train()

    # 3. both curves in one chart, on a log scale
    plt.plot(range(1, 31), train_losses)
    plt.plot(range(1, 31), val_losses, "--")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    # 4. the best epoch: the lowest validation loss
    best = np.argmin(val_losses)
    print(best + 1, val_losses[best], train_losses[best])
    print(train_losses[-1], val_losses[-1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The solid curve, the training loss, falls all the way: from 0.98 after the first epoch to 0.0055 after the last. The dashed curve, the validation loss, falls to its lowest point after epoch 8, at 0.41, with a training loss of 0.08 then, and rises to 0.48 by epoch 30. `np.argmin` gives the place of the lowest validation loss, counted from 0, so the epoch is `best + 1`; in the chart, `range(1, 31)` numbers the epochs from 1 to 30 along the x-axis. After epoch 8, the model gets better at its 1,000 digits and a little worse at new ones: overfitting. If you had kept the model of epoch 8, it would do best on new digits; the next lessons show how.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_diagnose():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Diagnose the curves

    The same MLP, now trained on only 300 training digits, with $\eta$ = 0.5, for 30 epochs, and measured on the same 500 validation digits. One run.
    """)
    return


@app.cell(hide_code=True)
def choice_diagnose_cell():
    _epoch, _loss = C.best(C.DIAGNOSE)
    _best_1000 = C.best("overfitting")[1]
    choice_diagnose = L.choice(
        "diagnose",
        question="What do these curves show, and the model of which epoch would you keep?",
        options={
            "right": "Overfitting; the model of the epoch where the dashed curve is lowest",
            "last": "Still learning; the model of epoch 30, where the solid curve is lowest",
            "stuck": "Stuck; every epoch is about as good as the others",
        },
        correct="right",
        why={
            "last": "The solid curve is the training loss: on the digits it learns from, the model keeps getting better. On new digits, it does not. See *Overfitting* in *The idea*.",
            "stuck": "The training loss falls from above 1 to close to 0.01: the steps do change the model. See *Stuck* in *The idea*.",
            "right": f"The training loss falls to close to 0, while the validation loss is lowest after epoch {_epoch}, at {L.number(_loss, 2)}, and then rises. With only 300 digits, that lowest point is higher than with the 1,000 digits of the worked example, {L.number(_best_1000, 2)}: fewer examples are easier to learn by heart.",
        },
    )
    mo.vstack([
        mo.Html(C.diagnose_picture({"title": "300 training digits, η = 0.5", "key": "training loss solid, validation loss dashed",
                                    "epoch": "epoch", "loss": "loss",
                                    "alt": f"The loss per epoch on a log scale. The training loss, solid, falls from above 1 to close to 0.01. The validation loss, dashed, falls to its lowest point after epoch {_epoch}, at {L.number(_loss, 2)}, and then rises slowly to about 0.76."})),
        choice_diagnose,
    ])
    return (choice_diagnose,)


@app.cell(hide_code=True)
def text_exercise_curves():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Measure both curves

        The run of exercise 1, with your own `DigitsMLP` and `fit` from your project. Each epoch, `fit(model, loader, 1, 0.5)` trains one epoch. Replace the `...` with step 2 of the recipe: after the epoch, measure the loss on all of `X_train` and on all of `X_val`, in eval mode and without a graph, and append each to its list. The last line shows the last training loss, the last validation loss and the lowest validation loss.
        """),
        L.course_versions(),
        L.task("After every epoch, append the training loss to `train_losses` and the validation loss to `val_losses`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_curves():
    box_curves = L.code_box("curves", lines=26)
    box_curves
    return (box_curves,)


@app.cell(hide_code=True)
def check_curves(box_curves):
    result_curves = L.check("curves", box_curves.value)
    result_curves
    return (result_curves,)


@app.cell(hide_code=True)
def hints_curves():
    L.hints(
        "curves",
        [
            "Step 2 of the recipe, as in the worked example, indented under `for epoch in range(30):`, after the line with `fit`.",
            "Eval mode, then a `with torch.no_grad():` block with two `append` lines: `loss_fn` on the model's logits for all of `X_train` with `y_train`, and the same for `X_val` with `y_val`. Then back to train mode.",
            "`model.eval()`, then `with torch.no_grad():` with `train_losses.append(loss_fn(model(___), ___).item())` and `val_losses.append(...)` for `X_val`, then `model.train()`.",
        ],
        why="After each epoch, `loss_fn(model(X_train), y_train)` is the training loss and `loss_fn(model(X_val), y_val)` the validation loss; `.item()` makes each a number for the list. The last training loss is 0.0098, close to 0, while the validation loss ends at 0.7589, above its lowest point, 0.6424.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_remedy():
    mo.md(r"""
    ### Exercise 3. What helps?

    Five things you can do after reading the curves. Put each with the curves whose problem it fixes: still learning, overfitting or stuck.
    """)
    return


@app.cell(hide_code=True)
def exercise_remedy():
    sort_remedy = L.sort("remedy")
    sort_remedy
    return (sort_remedy,)


@app.cell(hide_code=True)
def check_remedy(sort_remedy):
    result_remedy = L.check("remedy", sort_remedy)
    result_remedy
    return (result_remedy,)


@app.cell(hide_code=True)
def hints_remedy():
    L.hints(
        "remedy",
        [
            "For each remedy, ask what it changes: how long the model trains, which model you keep, what it learns from, or how large its steps are.",
            "*Overfitting* gets two remedies, and so does *Stuck*.",
            "Training longer is for curves that had come down a long way and were still coming down. Which two remedies fight learning by heart, and which two make the steps do something?",
        ],
        why="**Still learning:** train for more epochs. **Overfitting:** keep the model of the epoch with the lowest validation loss, and train on more examples. **Stuck:** raise the learning rate, and check that each update step changes the parameters.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The validation set of [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) now gives a curve, epoch by epoch, next to the training loss your `fit` from [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) already gave. The overfitting of [Overfitting and underfitting](/course/open/ml-overfitting?lang=en) shows up in it as a validation loss that rises again, and a log scale shows what happens close to 0. To keep the model of the best epoch, you have to store it while training goes on: in the next lesson, *Saving and loading a model*, you learn how.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_diagnose, result_curves, result_remedy):
    L.wrap_up(choice_diagnose, result_curves, result_remedy)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
