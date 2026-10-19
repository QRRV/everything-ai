# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-early-stopping. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-early-stopping", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading loss curves](/course/open/torch-loss-curves?lang=en) and [Saving and loading a model](/course/open/torch-checkpoints?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_jump():
    quick_jump = L.choice(
        "quick-jump",
        question="The validation loss of a run is 0.43, 0.41, 0.44, 0.42, 0.40, 0.42 in six epochs in a row. What does the 0.44 after the 0.41 tell you?",
        options={
            "right": "Little by itself: one epoch up can be chance",
            "overfit": "That the model starts to overfit there",
            "stuck": "That the model is stuck",
        },
        correct="right",
        why={
            "overfit": "Two epochs later the loss is lower than ever. Look at the trend over several epochs. See *Watch out* in [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "stuck": "The loss still goes down over these epochs, from 0.43 to 0.40. See *Stuck* in [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "A validation set of a few hundred examples makes its loss jump a little by chance. Here it goes on down to 0.40 two epochs later.",
        },
    )
    quick_jump
    return (quick_jump,)


@app.cell(hide_code=True)
def quick_check_copy():
    quick_copy = L.choice(
        "quick-copy",
        question="To keep the parameters of one epoch while training goes on, what do you do?",
        options={
            "name": "`best = model.state_dict()`",
            "right": "`torch.save(model.state_dict(), path)`",
            "model": "`best = model`",
        },
        correct="right",
        why={
            "name": "That dictionary holds the model's own tensors, which training goes on changing. See *Watch out* in [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "model": "`best` is then the same model under another name, and it changes along with it. See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "right": "The file holds the numbers of that moment; later steps change the model, not the file.",
        },
    )
    quick_copy
    return (quick_copy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Saving and loading a model](/course/open/torch-checkpoints?lang=en) you trained 30 epochs and kept the epoch with the lowest validation loss. But how many epochs should you train? Too few, and the model is still learning when you stop; too many, and you spend time on epochs that only make it worse on new data. You do not know the number in advance, and it differs for every model and every dataset. Better to let the validation loss decide while training runs.

    **An everyday example.** You are looking for a parking space in a long street. Each space you pass is a little further from where you need to be, or a little closer. You cannot see the whole street at once, so you do not drive to the very end: once you have passed several spaces that are no better than the best one so far, you turn back to that one.
    """)
    return


@app.cell(hide_code=True)
def predict_stop():
    _end, _epoch, _loss = C.stop(1)
    guess_stop = L.guess(
        "patience-one",
        question="The run of [Reading loss curves](/course/open/torch-loss-curves?lang=en) has its lowest validation loss, 0.41, after epoch 8. You stop training as soon as one epoch is not better than the best so far, and keep the best epoch until then. Which epoch do you think you keep?",
        options={
            "eight": "Epoch 8, with 0.41",
            "thirty": "Epoch 30",
            "right": "An earlier epoch, with a higher loss",
        },
        reveal=f"**An earlier epoch, with a higher loss.** The validation loss goes 1.09, 0.65, 0.53, 0.45, and then up to 0.51 by chance after epoch 5. So training stops after epoch {_end}, and keeps epoch {_epoch}, with {L.number(_loss, 2)}: the lowest point after epoch 8 is never reached. One epoch up is not enough to stop on. Waiting a few epochs, the *patience* of *The idea*, gets you past such a jump.",
    )
    guess_stop
    return (guess_stop,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Early stopping** stops training once the validation loss has not improved for a while, and keeps the checkpoint of the best epoch. After every epoch:

        - Is the validation loss the lowest so far? Then save a checkpoint, remember the epoch, and set the count of epochs without improvement back to 0.
        - If not, add 1 to that count. Once it reaches the **patience**, a number you choose, stop. With `break`, as in [Repeating until something happens: while](/course/open/py-while?lang=en), the loop ends at once.

        Afterwards you load the checkpoint: the model of the best epoch, not the model at the stop. That one is already `patience` epochs further, and no better on the validation set.

        The patience is a hyperparameter, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en). Too small, and a jump by chance stops training too early, as in the guess. Too large, and you train many epochs that do not help. A validation loss that jumps a lot, from a small validation set, needs more patience.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **early stopping** | | stopping training once the validation loss has not improved for `patience` epochs, and keeping the checkpoint of the best epoch | `break` |
        | **patience** | | how many epochs in a row without a new lowest validation loss you wait before you stop | `patience = 5` |
        """),
        L.watch_out("Early stopping chooses the epoch with the validation set. So the validation loss of that epoch is a little too good: it is the lowest of many tries. For an honest number on new data, measure the kept model once more, on the test set."),
        mo.md(r"""
        The picture: the run of [Reading loss curves](/course/open/torch-loss-curves?lang=en), with early stopping at a patience of 5. One run, the training loss solid, the validation loss dashed.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_stop():
    _end, _epoch, _loss = C.stop(5)
    mo.vstack([
        mo.Html(C.picture(5, {"title": "Early stopping with a patience of 5", "epoch": "epoch", "loss": "loss",
                              "kept": "kept", "stop": "stop",
                              "alt": f"The training loss, solid, and the validation loss, dashed, on a log scale, up to epoch {_end}. A dot marks the lowest validation loss, after epoch {_epoch}, at {L.number(_loss, 2)}; a thin line marks the stop, after epoch {_end}."})),
        mo.md(f"""
        The validation loss is lowest after epoch {_epoch}, at <span class="c-loss ea-num">{L.number(_loss, 2)}</span>. In the five epochs after it, up to epoch {_end}, none was lower, so training stops there, and the checkpoint of epoch {_epoch} is the model you keep. The 30 epochs of [Reading loss curves](/course/open/torch-loss-curves?lang=en) were {30 - _end} more than needed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train with early stopping:

    1. Choose the patience, and start with `waited = 0`.
    2. After every epoch, measure the validation loss.
    3. Is it the lowest so far? Save a checkpoint, remember the epoch, and set `waited` back to 0.
    4. If not, add 1 to `waited`; once `waited` reaches the patience, stop with `break`.
    5. After the loop, load the checkpoint into a new model: that is the model you keep.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The run of the picture, with the course's own copy of your `DigitsMLP`: 1,000 training digits, 500 validation digits, $\eta$ = 0.5, at most 30 epochs, a patience of 5. The checkpoint goes into your work folder, as in the lesson before. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_stop():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    X_train, y_train = X[:1000], y[:1000]
    X_val, y_val = X[5000:5500], y[5000:5500]
    loss_fn = nn.CrossEntropyLoss()
    path = C.checkpoint_path("early.pt")

    torch.manual_seed(0)
    model = DigitsMLP(64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)
    torch.manual_seed(1)
    # 1. the patience, and no epochs waited yet
    patience = 5
    waited = 0
    best_loss = 100.0   # higher than any loss here, so the first epoch is a new lowest
    best_epoch = 0
    for epoch in range(30):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 2. the validation loss after the epoch
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        model.train()
        # 3. a new lowest: save, remember, start counting again
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), path)
            waited = 0
        # 4. otherwise count, and stop at the patience
        else:
            waited = waited + 1
            if waited == patience:
                break
    print(epoch + 1, best_epoch, best_loss)

    # 5. the model you keep: the checkpoint of the best epoch
    kept = DigitsMLP(64)
    kept.load_state_dict(torch.load(path))
    kept.eval()
    with torch.no_grad():
        print(loss_fn(kept(X_val), y_val).item(), loss_fn(model(X_val), y_val).item())
    return


@app.cell(hide_code=True)
def text_after_example():
    _end, _epoch, _loss = C.stop(5)
    _stop_loss = C.curves()[1][_end - 1]
    mo.md(f"""
    Training stops after epoch {_end}: `epoch` is then {_end - 1}, counted from 0. The best epoch is {_epoch}, with a validation loss of {L.number(_loss, 4)}, and the kept model, loaded from the file, has exactly that loss. The model at the stop is a little worse on the validation set, {L.number(_stop_loss, 4)}. `best_loss` starts at `100.0`, higher than any loss here, so that the first epoch is always a new lowest.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which model?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-model",
        question="Early stopping with a patience of 5 stops after epoch 13; the lowest validation loss was after epoch 8. Which model do you use?",
        options={
            "stop": "The model after epoch 13: it trained the longest",
            "right": "The checkpoint of epoch 8",
            "retrain": "None yet: first train 13 epochs more",
        },
        correct="right",
        why={
            "stop": "After epoch 8, the validation loss did not get lower in five epochs: the model at the stop is no better on the validation set. See *The idea*.",
            "retrain": "Training on would only go further past the best epoch. See *The idea*.",
            "right": "The checkpoint holds the parameters of epoch 8, the best on the validation set. The five epochs after it only served to be sure that no better one came.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_stop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Let it stop by itself

        Your `DigitsMLP` and `fit`, on 300 training digits, with a patience of 5 and at most 100 epochs. After every epoch, the validation loss goes into `val_losses`. Replace the `...` with steps 3 and 4 of the recipe: at a new lowest validation loss, save the parameters to `path` and remember the epoch, counted from 1, in `best_epoch`; otherwise count the epoch in `waited`, and stop with `break` when it reaches `patience`. The test for a new lowest is `val_losses[-1] == min(val_losses)`, as in [Saving and loading a model](/course/open/torch-checkpoints?lang=en). The last line shows the epoch after which training stopped, the epoch you keep, and that model's validation loss.
        """),
        L.course_versions(),
        L.task("Save at every new lowest validation loss, count the epochs without one, and stop when the count reaches `patience`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stop():
    box_stop = L.code_box("early-stop", lines=38)
    box_stop
    return (box_stop,)


@app.cell(hide_code=True)
def check_stop(box_stop):
    result_stop = L.check("early-stop", box_stop.value)
    result_stop
    return (result_stop,)


@app.cell(hide_code=True)
def hints_stop():
    L.hints(
        "early-stop",
        [
            "Steps 3 and 4 of the recipe, as in the worked example; only the test for a new lowest is the one of the lesson before.",
            "An `if` with the test for a new lowest, and an `else`. In the `if`: save, remember the epoch, set `waited` to 0. In the `else`: add 1 to `waited`, and an `if` inside it that stops the loop.",
            "`if val_losses[-1] == min(val_losses):` with `torch.save(...)`, `best_epoch = epoch + 1` and `waited = 0`; `else:` with `waited = waited + 1` and `if waited == patience:` with `___`.",
        ],
        why="The validation loss is lowest after epoch 9. Epochs 10 to 14 are not lower, so `waited` goes from 1 to 5, and after epoch 14 `break` ends the loop. The file holds epoch 9, so the kept model has 0.6424, while the model at the stop has a higher one, 0.6998.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_patience():
    mo.md(r"""
    ### Exercise 3. How much patience?
    """)
    return


@app.cell(hide_code=True)
def choice_patience_cell():
    choice_patience = L.choice(
        "patience",
        question="Your validation set has only 100 digits, so its loss jumps a lot from epoch to epoch. What patience do you choose, and what does it cost?",
        options={
            "small": "A small one, such as 1: then a jump stops training before the model overfits",
            "none": "It does not matter: early stopping always keeps the best epoch",
            "right": "A larger one, such as 10: training takes more epochs after the best one",
        },
        correct="right",
        why={
            "small": "A jump by chance is not overfitting; with a patience of 1 it stops training too early, as in the guess. See *The idea*.",
            "none": "It keeps the best epoch until the stop. Stop too early, and the best epoch never comes. See the guess.",
            "right": "With more patience, a jump by chance does not stop training. The cost: after the best epoch, it still trains `patience` epochs before it stops.",
        },
    )
    choice_patience
    return (choice_patience,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Early stopping reads the curves of [Reading loss curves](/course/open/torch-loss-curves?lang=en) while training runs, and uses the checkpoints of [Saving and loading a model](/course/open/torch-checkpoints?lang=en) to keep the best epoch. It is a simple brake on overfitting: training stops soon after the model starts to learn its examples by heart. In the next lesson, *Data augmentation*, you fight overfitting from the other side: with more varied examples, made from the ones you have.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_patience, choice_which, result_stop):
    L.wrap_up(choice_which, result_stop, choice_patience)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
