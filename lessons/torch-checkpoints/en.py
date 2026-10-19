# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-checkpoints. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-checkpoints", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) and [Reading loss curves](/course/open/torch-loss-curves?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_best():
    quick_best = L.choice(
        "quick-best",
        question="During training, the validation loss after each epoch is 0.90, 0.61, 0.50, 0.52, 0.55 and 0.60, while the training loss keeps falling. After which epoch was the model best on new data?",
        options={
            "last": "Epoch 6: there the training loss is lowest",
            "right": "Epoch 3",
            "first": "Epoch 1: after that it only learns by heart",
        },
        correct="right",
        why={
            "last": "The training loss says how well the model fits the examples it learns from, not new ones. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "first": "The validation loss still falls after epoch 1, to 0.50. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "The validation loss is lowest after epoch 3, at 0.50, and rises after that: from there on, the model overfits.",
        },
    )
    quick_best
    return (quick_best,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="What does your `fit(model, loader, epochs, lr)` from [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) give back?",
        options={
            "model": "A new, trained model",
            "nothing": "Nothing: it only trains",
            "right": "The mean loss of each epoch",
        },
        correct="right",
        why={
            "model": "`fit` trains the model it gets: `model` itself changes. See exercise 3 of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "nothing": "Look at its last line, `return losses`. See exercise 3 of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "right": "It trains `model` itself, and gives back a list with the mean loss of the mini-batches of each epoch.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A trained model lives in the memory of the program that trained it. Stop the program, and the model is gone: the demo of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) trains your MLP again every time. And [Reading loss curves](/course/open/torch-loss-curves?lang=en) showed that the best model is often not the last one, but one from an epoch in the middle. Both ask for the same thing: the parameters of a model, kept in a file, to load again later.

    **An everyday example.** A long computer game lets you save. You stop, and tomorrow you go on from where you were, with everything you had. And before a hard part you save too: if it goes wrong, you load the save and are back where things were still good.
    """)
    return


@app.cell(hide_code=True)
def predict_live():
    guess_live = L.guess(
        "live",
        question="An MLP trains for 30 epochs. After epoch 8, where the validation loss is lowest, you write `best = model.state_dict()`, and training goes on to epoch 30. Then a new model gets `best` with `load_state_dict`. Which model do you think you get?",
        options={
            "eight": "The model of epoch 8",
            "right": "The model of epoch 30",
            "error": "An error: `best` is gone once training goes on",
        },
        reveal="**The model of epoch 30.** `model.state_dict()` copies nothing: its dictionary holds the model's own tensors, and every step after epoch 8 changes them. In the run of the worked example of [Reading loss curves](/course/open/torch-loss-curves?lang=en), the new model has a validation loss of 0.48, that of epoch 30, not the 0.41 of epoch 8. A file written with `torch.save` after epoch 8 does keep epoch 8: you do that in exercise 2.",
    )
    guess_live
    return (guess_live,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The state_dict.** `model.state_dict()` is a dictionary with every parameter of the model by name. The names come from the attributes in `__init__`: `self.hidden` gives `hidden.weight` and `hidden.bias`, `self.out` gives `out.weight` and `out.bias`. Each value is the tensor itself.

        **Save and load.** `torch.save(obj, path)` writes tensors, and dictionaries and lists of them, to a file; such a file usually ends in `.pt`. `torch.load(path)` reads it back. `model.load_state_dict(state)` copies the tensors of a state_dict into a model. That model must be of the same class, with the same sizes: every name must be there, with the same shape, or PyTorch stops with an error. The file holds only names and numbers; the code of the class stays in your own files, such as `nn/torch_mlp.py` in your project.

        **A checkpoint** is what you save while training: at least the state_dict of the model. To train on from there later, it also holds the epoch and the optimizer's own state_dict, `opt.state_dict()`, all in one dictionary. For plain SGD, that state_dict holds only its settings, such as the learning rate; optimizers that you meet later also keep numbers per parameter in it.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **state_dict** | state dict | the dictionary of PyTorch with every parameter of a model by name | `model.state_dict()` |
        | `torch.save(obj, path)` | torch save | writes tensors, and dictionaries and lists of them, to a file | `torch.save(model.state_dict(), path)` |
        | `torch.load(path)` | torch load | reads such a file back | `torch.load(path)` |
        | `load_state_dict(state)` | load state dict | copies the tensors of a state_dict into a model of the same class and sizes | `model.load_state_dict(state)` |
        | **checkpoint** | | a saved copy of the parameters at one moment of training, often with the epoch and the optimizer | `torch.save({"epoch": 10, ...}, path)` |
        """),
        L.watch_out("`model.state_dict()` copies nothing: the tensors in it are the model's own, and training goes on changing them. To keep the parameters of one moment, write them to a file with `torch.save`. And `torch.load` reads back only tensors and simple values, such as numbers and dictionaries. That is on purpose: a file from someone else could otherwise run code on your computer when you load it."),
        mo.md(r"""
        The picture: the state_dict of `DigitsMLP(64)`, the name of each parameter with the shape of its tensor.
        """),
        draw.dict_boxes(C.state_pairs(), key_label="name", value_label="tensor", title="model.state_dict() of DigitsMLP(64)"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To keep a model in a file and get it back:

    1. Save its parameters: `torch.save(model.state_dict(), path)`.
    2. Load them into a new model of the same class and sizes: `model = DigitsMLP(64)`, then `model.load_state_dict(torch.load(path))`. To predict, switch it to eval mode.
    3. To train on later, save a checkpoint: a dictionary with the epoch, `model.state_dict()` and `opt.state_dict()`. Load each part into a new model and a new optimizer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The MLP of the demo of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en): 64 hidden neurons, 10 epochs on the 6,000 training digits with $\eta$ = 0.5. The cell saves it, loads it into a new model, measures that one on the 1,000 test digits, and then saves and loads a checkpoint. `C.checkpoint_path("digits.pt")` gives the path of a file in your work folder, `work/checkpoints/digits.pt`. The comments with `#` say which step of the recipe each part is. Run the cell; it takes a second or two.
    """)
    return


@app.cell
def example_save():
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
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(data["test_labels"], dtype=torch.int64)
    torch.manual_seed(0)
    model = DigitsMLP(64)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    torch.manual_seed(2)
    for epoch in range(10):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 1. save the parameters
    state = model.state_dict()
    for name in state:
        print(name, state[name].shape)
    path = C.checkpoint_path("digits.pt")
    torch.save(model.state_dict(), path)

    # 2. load them into a new model of the same class and sizes
    loaded = DigitsMLP(64)
    print(loaded.load_state_dict(torch.load(path)))
    loaded.eval()
    with torch.no_grad():
        pred = loaded(X_test).argmax(dim=1)
    print((pred == y_test).float().mean().item())

    # 3. a checkpoint to train on later: the epoch, the model and the optimizer
    torch.save({"epoch": 10, "model": model.state_dict(), "opt": opt.state_dict()}, C.checkpoint_path("digits-epoch10.pt"))
    checkpoint = torch.load(C.checkpoint_path("digits-epoch10.pt"))
    again = DigitsMLP(64)
    again.load_state_dict(checkpoint["model"])
    opt_again = torch.optim.SGD(again.parameters(), lr=0.5)
    opt_again.load_state_dict(checkpoint["opt"])
    print(checkpoint["epoch"], checkpoint["opt"]["param_groups"][0]["lr"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The state_dict has four tensors, with the shapes of the picture: 784 · 64 + 64 + 64 · 10 + 10 = 50,890 numbers. The file is about 200 kB: a float32 takes 32 bits, which is 4 bytes, and the names take a little more. `load_state_dict` reports `<All keys matched successfully>`: every name of the file was found in the new model, with the right shape. The loaded model reads 929 of the 1,000 test digits right, exactly as in the demo of the project lesson: the same numbers give the same model. The checkpoint gives back the epoch, 10, and the settings of the optimizer, such as its learning rate, 0.5; with those, a loop could go on training `again` from epoch 11.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_saved():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What is in the file?
    """)
    return


@app.cell(hide_code=True)
def choice_saved_cell():
    choice_saved = L.choice(
        "what-saved",
        question="What does `torch.save(model.state_dict(), path)` put in the file?",
        options={
            "code": "The class `DigitsMLP` with its code, and its parameters",
            "right": "The parameters of each layer, by name",
            "preds": "The predictions of the model for the training digits",
        },
        correct="right",
        why={
            "code": "The code stays in your own files: to load, you make a `DigitsMLP` first. See *Save and load* in *The idea*.",
            "preds": "A state_dict holds what the model learned, not what it predicted. See *The state_dict* in *The idea*.",
            "right": "A state_dict holds every parameter by name: `hidden.weight`, `hidden.bias`, `out.weight` and `out.bias`. Loading it needs a model of the same class and sizes.",
        },
    )
    choice_saved
    return (choice_saved,)


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Keep the best epoch

        The run of exercise 1 of [Reading loss curves](/course/open/torch-loss-curves?lang=en): your `DigitsMLP` and `fit`, 300 training digits, 30 epochs. After every epoch, the validation loss goes into `val_losses`, and the `if` is true when it is the lowest so far. Replace the two `...`: inside the `if`, save the parameters of the model to `path`; after the loop, load the file into `best`. In this box, `checkpoint_path("best.pt")` gives a path in the check's own folder, `work/checkpoints/check/`, and first removes the file of an earlier run, so an old file never gets in the way. The last line shows the validation loss of the model after epoch 30, and that of `best`.
        """),
        L.course_versions(),
        L.task("Save the parameters whenever the validation loss is the lowest so far, and load the last file you saved into `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-epoch", lines=36)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-epoch", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-epoch",
        [
            "Steps 1 and 2 of the recipe: one line inside the `if`, one line after the loop.",
            "Inside the `if`, write the state_dict of `model` to the file at `path`. After the loop, `best` is a new `DigitsMLP(64)`: give it what `torch.load` reads from that file.",
            "`torch.save(model.___(), path)` inside the `if`, and `best.___(torch.load(path))` after the loop.",
        ],
        why="Each time the validation loss is the lowest so far, `torch.save(model.state_dict(), path)` writes the parameters of that moment over the old file. The last time is after epoch 9, so the file keeps epoch 9 while training goes on. `best.load_state_dict(torch.load(path))` gives `best` those parameters: a validation loss of 0.6424, against 0.7589 for the model after epoch 30.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_checkpoint():
    mo.md(r"""
    ### Exercise 3. What goes in a checkpoint?

    Five things from a training run. Put each in its group: in every checkpoint, only when you want to train on later, or not in the checkpoint at all.
    """)
    return


@app.cell(hide_code=True)
def exercise_checkpoint():
    sort_checkpoint = L.sort("checkpoint")
    sort_checkpoint
    return (sort_checkpoint,)


@app.cell(hide_code=True)
def check_checkpoint(sort_checkpoint):
    result_checkpoint = L.check("checkpoint", sort_checkpoint)
    result_checkpoint
    return (result_checkpoint,)


@app.cell(hide_code=True)
def hints_checkpoint():
    L.hints(
        "checkpoint",
        [
            "For each thing, ask: does someone who only predicts need it, does someone who trains on need it, or is it kept somewhere else already?",
            "A checkpoint holds numbers that training changes. What does training change, and what not?",
            "The parameters are needed by everyone. The optimizer and the epoch only matter to train on. Which two things live in your files anyway?",
        ],
        why="**In every checkpoint:** `model.state_dict()`, the parameters. **Only to train on later:** `opt.state_dict()` and the number of the epoch. **Not in the checkpoint:** the code of `DigitsMLP`, which is in your project, and the training digits, which training does not change.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The parameters that [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en) made for every layer now have names in a state_dict, and `torch.save` keeps them in a file. The best epoch of [Reading loss curves](/course/open/torch-loss-curves?lang=en) is now something you can keep: write the parameters each time the validation loss is the lowest so far. In the next lesson, *Early stopping*, training stops by itself once the validation loss has not improved for a while, and you keep the best checkpoint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_saved, result_best, result_checkpoint):
    L.wrap_up(choice_saved, result_best, result_checkpoint)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
