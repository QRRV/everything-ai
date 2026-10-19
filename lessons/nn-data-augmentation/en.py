# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-data-augmentation. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-data-augmentation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en) and [Reading loss curves](/course/open/torch-loss-curves?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pixel():
    quick_pixel = L.choice(
        "quick-pixel",
        question="`img` is a grayscale image of 28 by 28 pixels, as an array of shape (28, 28). What is `img[3, 10]`?",
        options={
            "column": "The pixel in column 3, row 10",
            "right": "The pixel in row 3, column 10",
            "count": "The number of pixels in row 3 up to column 10",
        },
        correct="right",
        why={
            "column": "The first index counts the rows, from the top. See [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en).",
            "count": "An index gives one number, the brightness of one pixel. See [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en).",
            "right": "Axis 0 runs down the rows, axis 1 across the columns: `img[3, 10]` is the brightness of the pixel in row 3, column 10, both counted from 0.",
        },
    )
    quick_pixel
    return (quick_pixel,)


@app.cell(hide_code=True)
def quick_check_gap():
    quick_gap = L.choice(
        "quick-gap",
        question="A model's training loss keeps falling, but its validation loss rises again after epoch 12. Why does the validation loss rise?",
        options={
            "right": "It learns its training examples by heart",
            "large": "The learning rate is too large: the steps overshoot",
            "harder": "The validation digits are harder than the training digits",
        },
        correct="right",
        why={
            "large": "A learning rate that is too large makes the training loss jump or grow as well, and here it keeps falling. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "harder": "Harder digits would give a higher validation loss from the start, not one that first falls and then rises. See *Overfitting* in [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "Overfitting: after epoch 12, the model gets better at its own examples and worse at new ones. This lesson is about one way to make that harder.",
        },
    )
    quick_gap
    return (quick_gap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Reading loss curves](/course/open/torch-loss-curves?lang=en), an MLP started to learn its 1,000 training digits by heart after about 8 epochs. More training examples make that harder, but new examples cost time and money: someone has to write them and label them. Yet a digit written one pixel further to the right is still the same digit. Can you make new examples from the ones you have, for free?

    **An everyday example.** Someone learns to read a new handwriting from one page only. After a while they know that page by heart, word for word, yet they stumble on the next page, where the same letters sit a little higher or lower on the line, or a little further to the left. More pages in that handwriting would help, but there are none.
    """)
    return


@app.cell(hide_code=True)
def predict_shift():
    _epoch0, _loss0 = C.best(0)
    _epoch1, _loss1 = C.best(1)
    guess_shift = L.guess(
        "shift-loss",
        question=f"The run of [Reading loss curves](/course/open/torch-loss-curves?lang=en) reaches its lowest validation loss, {L.number(_loss0, 2)}, after epoch {_epoch0}. Now every time a training digit is used, it is shifted first, at random, by up to 1 pixel up or down and up to 1 pixel left or right. The rest stays the same. What do you think the lowest validation loss does?",
        options={
            "same": "It stays about the same: the digits are the same",
            "right": "It gets much lower",
            "higher": "It gets higher: the model learns from messier examples",
        },
        reveal=f"**It gets much lower.** With the shifts, the validation loss is lowest after epoch {_epoch1}, at {L.number(_loss1, 2)}, against {L.number(_loss0, 2)} without. The model sees each digit in up to nine positions, a newly drawn one every epoch, so it can hardly learn the exact pixels of its 1,000 examples by heart. It has to learn what a 3 looks like wherever it is written, and that works on new digits too.",
    )
    guess_shift
    return (guess_shift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Data augmentation** makes new training examples from the ones you have, with small changes that keep the label: a digit shifted by a pixel is still the same digit. The changes are random, and new every time an example is used, so in every epoch the model sees other variants. The model can hardly learn exact pixels by heart; it has to learn what stays the same.

        Where to do it: in the `__getitem__` of a Dataset, as in [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en). The DataLoader asks for every example once per epoch, so every epoch gets new random changes. Only the training set gets them: the validation set and the test set stay as they are, because they measure how the model does on real digits.

        For the digits, the change here is a shift. The digits are rows of 784 numbers, and `.reshape(28, 28)` makes an image of one, on a tensor as on a NumPy array. `torch.roll(img, shifts=(dy, dx), dims=(0, 1))` moves a 28 by 28 image `dy` rows down and `dx` columns to the right; negative numbers go up and to the left. What rolls off one side comes back on the other. Around most digits there is a border of empty paper, so only empty paper rolls around. Not around all: in 14 of the 1,000 training digits, mostly 7s, ink touches the edge, and in about 1 in 8 it comes within two pixels of it. A shift towards that edge brings a few pixels of ink back on the other side. The shift is drawn with `random.randint`, as in [Using code from modules](/course/open/py-imports?lang=en).

        Not every change keeps the label right. A digit turned upside down can become another digit: a 6 becomes a 9. Choose changes that real examples also have.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **data augmentation** | | new training examples from the ones you have, with random changes that keep the label | in `__getitem__` |
        | `torch.roll(t, shifts, dims)` | torch roll | moves the numbers of a tensor along the axes `dims`; what falls off one side comes back on the other | `torch.roll(img, shifts=(1, -1), dims=(0, 1))` |
        """),
        L.watch_out("Augment only the training set. A validation set with random changes measures something else every epoch, and a test set with them no longer says how the model does on the real digits it will meet."),
        mo.md(r"""
        The picture: training digit 0, a 5, as it is and shifted by one pixel four ways, with its shift (down, across) below.
        """),
        mo.Html(C.variants_picture({"title": "One digit, four shifts", "original": "as it is",
                                    "alt": "Five images of the same handwritten 5: as it is, and shifted one pixel up, down, left and right. Each is clearly the same digit."})),
    ])
    return


@app.cell(hide_code=True)
def picture_curves():
    _epoch0, _loss0 = C.best(0)
    _epoch1, _loss1 = C.best(1)
    mo.vstack([
        mo.md(r"""
        The effect: the validation loss per epoch of the MLP of [Reading loss curves](/course/open/torch-loss-curves?lang=en), on the same 1,000 training digits, without shifts and with shifts of up to 1 pixel. One run each: in four runs with other seeds, the lowest validation loss with shifts lay between 0.25 and 0.29, and without shifts at about 0.41 every time.
        """),
        mo.Html(C.curves_picture({"title": "Validation loss, with and without shifts", "epoch": "epoch", "loss": "validation loss",
                                  "plain": "without shifts", "shifted": "with shifts",
                                  "alt": f"Two dashed lines over 40 epochs. Without shifts, the validation loss is lowest after epoch {_epoch0}, at {L.number(_loss0, 2)}, and then rises. With shifts, it keeps falling longer, to {L.number(_loss1, 2)} after epoch {_epoch1}, and stays low."})),
        mo.md(f"""
        Without shifts, the validation loss is lowest after epoch {_epoch0}, at <span class="c-loss ea-num">{L.number(_loss0, 2)}</span>, and rises after that. With shifts, it goes on falling until epoch {_epoch1}, to <span class="c-loss ea-num">{L.number(_loss1, 2)}</span>. Learning takes longer, because every epoch brings new variants, but the model gets much better on digits it never saw.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train with data augmentation:

    1. Choose changes that keep the label, such as a shift of up to `k` pixels.
    2. Write a Dataset whose `__getitem__` draws a new random change every time, and applies it to the example.
    3. Use that Dataset for the training set only; the validation set and the test set stay as they are.
    4. Train as before, and compare the validation loss with a run without data augmentation.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A Dataset that shifts every digit by up to 1 pixel, the same as `C.ShiftedDigits` of the picture. First it gives the same example twice, to show that the shift is new each time; `C.shift_of(x, original)` gives the shift it finds, down and across. Then the cell trains the MLP of [Reading loss curves](/course/open/torch-loss-curves?lang=en) with it for 40 epochs. The comments with `#` say which step of the recipe each part is. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def example_augment():
    import random
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    # 1. the change: a shift of up to k pixels, down and across
    # 2. a Dataset that draws a new shift every time it gives an example
    class ShiftedDigits(Dataset):
        def __init__(self, images, labels, k):
            self.images = images
            self.labels = labels
            self.k = k

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            img = self.images[i].reshape(28, 28)
            dy = random.randint(-self.k, self.k)
            dx = random.randint(-self.k, self.k)
            img = torch.roll(img, shifts=(dy, dx), dims=(0, 1))
            return img.reshape(784), self.labels[i]

    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    # 3. shifts for the training set only; the validation set stays as it is
    train_set = ShiftedDigits(X[:1000], y[:1000], 1)
    X_val, y_val = X[5000:5500], y[5000:5500]

    random.seed(0)
    print(C.shift_of(train_set[0][0], X[0]), C.shift_of(train_set[0][0], X[0]))

    # 4. train as before, and measure the validation loss after each epoch
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    random.seed(0)
    model = DigitsMLP(64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(train_set, batch_size=50, shuffle=True)
    torch.manual_seed(1)
    val_losses = []
    for epoch in range(40):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val), y_val).item())
        model.train()
    best = np.argmin(val_losses)
    print(best + 1, val_losses[best], val_losses[-1])
    return


@app.cell(hide_code=True)
def text_after_example():
    _epoch1, _loss1 = C.best(1)
    _epoch0, _loss0 = C.best(0)
    mo.md(f"""
    The same example comes back with two different shifts: first (0, 0), one of the nine possible shifts, which leaves it as it is, then (−1, 0), one pixel up. The validation loss is lowest after epoch {_epoch1}, at {L.number(_loss1, 4)}, as in the picture, against {L.number(_loss0, 2)} without shifts. And after 40 epochs it is still low: the model hardly overfits. The 1,000 training digits did not change; only the way the model sees them did.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which set?
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where",
        question="You train with shifted digits. Which sets get the random shifts?",
        options={
            "all": "All three: training, validation and test, so they look alike",
            "two": "The training set and the validation set",
            "right": "Only the training set",
        },
        correct="right",
        why={
            "all": "The test set says how the model does on real digits; with random shifts it measures something else. See *Watch out* in *The idea*.",
            "two": "A validation set with random shifts gives another loss every epoch, even for the same model. See *Watch out* in *The idea*.",
            "right": "The shifts give the model more varied examples to learn from. The validation and test sets measure, and stay the same every time.",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_exercise_shifted():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A Dataset that shifts

        `ShiftedDigits(images, labels, k)` gives each digit shifted by up to `k` pixels. `__getitem__` already draws `dy`, the shift down. Replace the `...`: draw `dx` in the same way, and shift `img` by `dy` down and `dx` across. The last lines ask for example 2 twice, with `random.seed(3)` and `k` = 2, and show its label and the two shifts that `shift_of` finds, down and across.
        """),
        L.task("Draw `dx`, and shift `img` with `torch.roll` by `dy` down and `dx` across."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shifted():
    box_shifted = L.code_box("shifted", lines=26)
    box_shifted
    return (box_shifted,)


@app.cell(hide_code=True)
def check_shifted(box_shifted):
    result_shifted = L.check("shifted", box_shifted.value)
    result_shifted
    return (result_shifted,)


@app.cell(hide_code=True)
def hints_shifted():
    L.hints(
        "shifted",
        [
            "Step 2 of the recipe, as in `__getitem__` of the worked example.",
            "`dx` comes from `random.randint` with the same two numbers as `dy`. Then `torch.roll` gets `img`, the two shifts and the two axes, rows first.",
            "`dx = random.randint(-self.k, self.___)`, then `img = torch.roll(img, shifts=(___, ___), dims=(0, 1))`.",
        ],
        why="With `random.seed(3)` and `k` = 2, the first call draws a shift of −1 down, so 1 up, and 2 to the right; the second call draws 2 down and 1 to the left. The label stays the 4 of example 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_keeps():
    mo.md(r"""
    ### Exercise 3. Which changes keep the digit?

    Six changes you could make to a training digit. Put each in its group: keeps the digit, or can change the digit or make it unreadable.
    """)
    return


@app.cell(hide_code=True)
def exercise_keeps():
    sort_keeps = L.sort("keeps-label")
    sort_keeps
    return (sort_keeps,)


@app.cell(hide_code=True)
def check_keeps(sort_keeps):
    result_keeps = L.check("keeps-label", sort_keeps)
    result_keeps
    return (result_keeps,)


@app.cell(hide_code=True)
def hints_keeps():
    L.hints(
        "keeps-label",
        [
            "For each change, ask: do real handwritten digits differ in this way from each other?",
            "Three changes keep the digit, three can break it. Think of a 2, a 3, a 6 and a 9.",
            "A shift, a small turn and lighter ink keep the digit. Which three changes can turn a 6 into a 9, a 3 into something that is no digit, or an 8 into what could be a 9?",
        ],
        why="**Keeps the digit:** a shift of a pixel, a turn of 10 degrees, lighter ink. **Can change it or make it unreadable:** a mirror image, upside down, and cutting off the bottom half.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Data augmentation lives in the `__getitem__` of [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en), and works on images as the grids of numbers of [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en). The overfitting of [Reading loss curves](/course/open/torch-loss-curves?lang=en) comes later and is smaller, because the model can no longer learn exact pixels by heart. In the next lesson, *Using a pretrained model*, you do not train a digit reader yourself at all: you load one that someone else trained.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, result_keeps, result_shifted):
    L.wrap_up(choice_where, result_shifted, result_keeps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
