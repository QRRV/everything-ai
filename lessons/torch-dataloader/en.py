# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-dataloader. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-dataloader", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Mini-batches and epochs](/course/open/torch-batches?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_epoch():
    quick_epoch = L.choice(
        "quick-epoch",
        question="What is the difference between an iteration and an epoch?",
        options={
            "same": "None: two words for one pass over the data",
            "swapped": "An iteration passes over all the data; an epoch is one step",
            "right": "An iteration is one step on one mini-batch; an epoch, one pass over all data",
        },
        correct="right",
        why={
            "same": "An epoch holds many iterations: one for each mini-batch. See *The idea* in [Mini-batches and epochs](/course/open/torch-batches?lang=en).",
            "swapped": "It is the other way round. See the new words in [Mini-batches and epochs](/course/open/torch-batches?lang=en).",
            "right": "One epoch is as many iterations as there are mini-batches: 200 examples in batches of 20 give 10.",
        },
    )
    quick_epoch
    return (quick_epoch,)


@app.cell(hide_code=True)
def quick_check_used():
    quick_used = L.choice(
        "quick-used",
        question="`X` and `y` hold 200 examples, and `batches` is the generator of [Mini-batches and epochs](/course/open/torch-batches?lang=en). What does this show?\n\n```python\ng = batches(X, y, 20)\nfirst = 0\nfor xb, yb in g:\n    first = first + 1\nsecond = 0\nfor xb, yb in g:\n    second = second + 1\nprint(first, second)\n```",
        options={
            "twice": "`10 10`",
            "right": "`10 0`",
            "error": "Nothing: Python stops at the second loop",
        },
        correct="right",
        why={
            "twice": "`g` is one generator. After the first loop it has handed out all its batches, and a generator is used up after one pass. See the watch-out in [Generators: producing values one at a time](/course/open/py-generators?lang=en).",
            "error": "A used-up generator gives no error: a loop over it just runs zero times.",
            "right": "The first loop takes all 10 batches. After that `g` is used up, so the second loop gets nothing. For a new epoch, `batches(X, y, 20)` has to be called again.",
        },
    )
    quick_used
    return (quick_used,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your generator from [Mini-batches and epochs](/course/open/torch-batches?lang=en) works for data that fits in two tensors, and it hands out the batches in the same order every epoch. Real data is often a folder with thousands of images, or a file that you read piece by piece. And its order matters: data is often sorted, by digit or by date. PyTorch splits the job in two. A **Dataset** knows how many examples there are and how to get example number $i$. A **DataLoader** makes mini-batches from it, in a new random order every epoch.

    **An everyday example.** A new pack of cards comes sorted: first all the hearts, then the diamonds, and so on. Before every game you shuffle the pack, and then deal hands from it. The pack only holds the cards, each in its own place; the dealer makes the hands, from a new shuffle for every game.
    """)
    return


@app.cell(hide_code=True)
def predict_sorted():
    guess_sorted = L.guess(
        "sorted",
        question="Suppose the 6,000 digits were sorted by label: first all 592 zeros, then the ones, and so on. Mini-batches of 64 in that order, without shuffling. What do you think the first 9 mini-batches teach the model?",
        options={
            "sample": "A little of every digit: each mini-batch is a sample of the data",
            "right": "Only that the answer is 0: they hold nothing but zeros",
            "nothing": "Nothing yet: a model learns only at the end of an epoch",
        },
        reveal="**That the answer is always 0.** The 592 zeros fill the first 9 mini-batches of 64 (9 · 64 = 576), so for 9 steps every slope pulls toward \"always 0\". Then come only ones, and the steps pull toward \"always 1\", undoing the steps before. **Shuffling** mixes the digits in every mini-batch, so the slope of each step is a fair estimate of the slope over all the digits.",
    )
    guess_sorted
    return (guess_sorted,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A Dataset** is a class built on `Dataset`, from `torch.utils.data`, with two special methods, as in [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en):

        - **`__len__(self)`** gives back the number of examples: `len(ds)` runs it.
        - **`__getitem__(self, i)`** gives back example `i`, as the input and its label: `ds[i]` runs it.

        Its `__init__` gets the data ready, for instance as tensors. Data that is already in two tensors, with the examples along the first axis, needs no class of its own: `TensorDataset(X, y)` is such a Dataset.

        **A DataLoader** makes mini-batches from a Dataset: `loader = DataLoader(ds, batch_size=64, shuffle=True)`. Each `for xb, yb in loader:` is one epoch. The DataLoader asks the Dataset for examples, one by one, and stacks them into mini-batches: `xb` gets the batch as its first axis. With `shuffle=True` it **shuffles** them: every epoch a new random order, so every mini-batch is a new mix. The last mini-batch may be smaller, and `len(loader)` is the number of mini-batches. Unlike a generator, a DataLoader is not used up: each new loop over it is a new epoch. Shuffling uses PyTorch's random numbers, so `torch.manual_seed` before the loops gives the same shuffles on every run.

        | Your generator | A Dataset and a DataLoader |
        |---|---|
        | `batches(X, y, 20)`, called again for every epoch | `DataLoader(TensorDataset(X, y), batch_size=20)`, made once |
        | the same order every epoch | `shuffle=True`: a new order every epoch |
        | needs the data in two tensors | any data, through `__len__` and `__getitem__` |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Dataset** | | a class that says how many examples there are and gives example `i` | `class Digits(Dataset):` |
        | `__len__` | dunder len | the special method behind `len(ds)`: the number of examples | `def __len__(self):` |
        | `__getitem__` | dunder get item | the special method behind `ds[i]`: example `i`, as input and label | `def __getitem__(self, i):` |
        | `TensorDataset` | | a Dataset made from tensors whose first axis runs over the examples | `TensorDataset(X, y)` |
        | **DataLoader** | data loader | makes mini-batches from a Dataset; each loop over it is an epoch | `DataLoader(ds, batch_size=64, shuffle=True)` |
        | **shuffling** | | putting the examples in a new random order, every epoch | `shuffle=True` |
        """),
        L.watch_out("`ds[i]` is one example: an image of shape (784,) and its label. A mini-batch from the DataLoader has one more axis in front, the examples: `xb` has shape (64, 784), and `yb` shape (64,). A model gets mini-batches, so it always sees that extra axis."),
        mo.md(r"""
        The 6,000 digits sorted by label, as in the guess, and the labels of the first mini-batch of 64, the first 8 of them: without shuffling, and in two epochs of a DataLoader that shuffles, after `torch.manual_seed(0)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_shuffle():
    _grids = C.picture(["sorted", "epoch 1", "epoch 2"], ["only zeros, every epoch", "a random mix", "a new mix"])
    mo.vstack([draw.array_grids(_grids[:1], title="The first labels of the first mini-batch"),
               draw.array_grids(_grids[1:2]), draw.array_grids(_grids[2:])])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To hand out data with a DataLoader:

    1. Make a Dataset: a class on `Dataset` with `__init__`, `__len__` and `__getitem__`, or `TensorDataset(X, y)` for data in tensors.
    2. Make the DataLoader once: `loader = DataLoader(ds, batch_size=..., shuffle=True)`.
    3. Call `torch.manual_seed(...)` before the loops, for the same shuffles on every run.
    4. Loop over the epochs, and in each over `for xb, yb in loader:`, with an update step for each mini-batch.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The handwritten digits of [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en): 6,000 training images of QMNIST in `data/qmnist-small.npz`, each of 28 by 28 pixels as 784 numbers from 0 to 255, with its digit as the label. A Dataset for them, and a loop that counts the mini-batches of a DataLoader in 2 epochs, without training yet. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_dataloader():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    # 1. a Dataset for the digits
    class _Digits(Dataset):
        def __init__(self, images, labels):
            self.images = torch.tensor(images, dtype=torch.float32) / 255
            self.labels = torch.tensor(labels, dtype=torch.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return self.images[i], self.labels[i]

    _data = np.load("data/qmnist-small.npz")
    _ds = _Digits(_data["train_images"], _data["train_labels"])
    print(len(_ds), _ds[0][0].shape, _ds[0][1])
    # 2. the DataLoader, made once
    _loader = DataLoader(_ds, batch_size=64, shuffle=True)
    print(len(_loader))
    # 3. the same shuffles on every run
    torch.manual_seed(0)
    # 4. epochs, and mini-batches in each
    for _epoch in range(2):
        _batches = 0
        for _xb, _yb in _loader:
            _batches = _batches + 1
        print(_epoch, _batches, _xb.shape, _yb.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The Dataset holds 6,000 examples; example 0 is an image of 784 numbers, of the digit 5. The DataLoader makes 94 mini-batches of 64 per epoch: 93 full ones and a last one of 48, because 93 · 64 = 5,952. The loop ends with that last mini-batch, so `_xb` has shape (48, 784) and `_yb` shape (48,). The second epoch goes over the same DataLoader again, in a new order, without making anything new.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example_training():
    mo.md(r"""
    The same recipe for data that is already in two tensors, now with an update step for each mini-batch: the 200 examples and the line of [Mini-batches and epochs](/course/open/torch-batches?lang=en), for 2 epochs. Run the cell.
    """)
    return


@app.cell
def example_training(DataLoader, torch):
    from torch import nn
    from torch.utils.data import TensorDataset

    _X, _y = C.data()   # the 200 examples of the lesson before, shapes (200, 1) and (200, 1)
    # 1. a TensorDataset: the data is already in two tensors
    _ds = TensorDataset(_X, _y)
    # 2. the DataLoader, made once
    _loader = DataLoader(_ds, batch_size=20, shuffle=True)
    torch.manual_seed(1)
    _model = nn.Linear(1, 1)
    _opt = torch.optim.SGD(_model.parameters(), lr=0.1)
    # 3. the same shuffles on every run
    torch.manual_seed(0)
    # 4. epochs, and an update step for each mini-batch
    for _epoch in range(2):
        for _xb, _yb in _loader:
            _loss = ((_model(_xb) - _yb) ** 2).mean()
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
        print(_epoch, ((_model(_X) - _y) ** 2).mean().item())
    return


@app.cell(hide_code=True)
def text_after_training():
    mo.md(r"""
    Each epoch takes 10 steps, one for each shuffled mini-batch of 20. After the first epoch the loss over all data is about 0.33, after the second about 0.28. The step is the same four lines as in [Mini-batches and epochs](/course/open/torch-batches?lang=en); only where the mini-batches come from has changed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reshuffle():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What shuffling changes
    """)
    return


@app.cell(hide_code=True)
def choice_reshuffle_cell():
    choice_reshuffle = L.choice(
        "reshuffle",
        question="`loader = DataLoader(ds, batch_size=64, shuffle=True)`. What changes from one epoch to the next?",
        options={
            "right": "Which examples share a mini-batch",
            "examples": "The examples themselves: each epoch gets other images",
            "once": "Nothing after the first epoch: the order is drawn once",
        },
        correct="right",
        why={
            "examples": "The Dataset stays the same; every epoch still has each of its examples once. See *A DataLoader* in *The idea*.",
            "once": "Each loop over the DataLoader draws a new order. Look at epoch 1 and epoch 2 in the picture.",
            "right": "Every epoch has all 6,000 examples once, in a new random order, so each mini-batch is a new mix.",
        },
    )
    choice_reshuffle
    return (choice_reshuffle,)


@app.cell(hide_code=True)
def text_exercise_digits():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A Dataset for the digits

        Finish the class `Digits`: `__init__` already turns the images and labels into tensors. Replace the two `...`: `__len__` gives back the number of examples, and `__getitem__` the image and the label of example `i`, in that order. The last lines take example 7 and show the number of examples, the length of the image, its digit and the sum of its pixels.
        """),
        L.task("Make `Digits` a Dataset that knows its length and gives example `i`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_digits():
    box_digits = L.code_box("digits", lines=20)
    box_digits
    return (box_digits,)


@app.cell(hide_code=True)
def check_digits(box_digits):
    result_digits = L.check("digits", box_digits.value)
    result_digits
    return (result_digits,)


@app.cell(hide_code=True)
def hints_digits():
    L.hints(
        "digits",
        [
            "Step 1 of the recipe, as in the worked example.",
            "`__len__` gives back the length of one of the two tensors that `__init__` made. `__getitem__` gives back two things, separated by a comma.",
            "`return len(self.___)` and `return self.images[i], ___`.",
        ],
        why="`self.labels` holds one label per example, so its length is 6,000. Example 7 is row 7 of `self.images` and number 7 of `self.labels`: an image of the digit 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shuffled():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Training with a DataLoader

        The data and the model of [Mini-batches and epochs](/course/open/torch-batches?lang=en), with an optimizer. Replace the two `...`: `loader` is a DataLoader that shuffles, over `TensorDataset(X, y)`, with mini-batches of 20, and each epoch loops over it with an update step for every mini-batch. The last line is the loss over all data after 3 epochs. The check also tries another learning rate.
        """),
        L.task("Train for 3 epochs with a DataLoader that shuffles."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shuffled():
    box_shuffled = L.code_box("shuffled", lines=20)
    box_shuffled
    return (box_shuffled,)


@app.cell(hide_code=True)
def check_shuffled(box_shuffled):
    result_shuffled = L.check("shuffled", box_shuffled.value)
    result_shuffled
    return (result_shuffled,)


@app.cell(hide_code=True)
def hints_shuffled():
    L.hints(
        "shuffled",
        [
            "Steps 1, 2 and 4 of the recipe, as in the second part of the worked example; the seed of step 3 is already there.",
            "The DataLoader gets the Dataset, the batch size and `shuffle`. Inside the epoch loop, a loop over `loader` takes the place of the generator of the lesson before.",
            "`loader = DataLoader(TensorDataset(X, y), batch_size=___, shuffle=___)` and `for xb, yb in ___:` with the four lines of an update step.",
        ],
        why="Every epoch the DataLoader hands out the 200 examples in 10 shuffled mini-batches of 20, each with its own step. After 3 epochs the loss is about 0.279.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A Dataset uses special methods, as in [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en): `len(ds)` and `ds[i]`. A DataLoader takes over the job of the generator of [Mini-batches and epochs](/course/open/torch-batches?lang=en), and shuffles, so every epoch is a new mix and the descent is truly stochastic. In the next lesson, *Training mode and evaluation mode*, you switch a model from learning to just predicting.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_reshuffle, result_digits, result_shuffled):
    L.wrap_up(choice_reshuffle, result_digits, result_shuffled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
