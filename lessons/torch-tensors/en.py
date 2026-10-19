# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-tensors. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-tensors", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en) and [Broadcasting](/course/open/np-broadcasting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_backward():
    quick_backward = L.choice(
        "quick-backward",
        question="Your `train` from [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en) trains `MLP(1, [8, 1])` on 12 examples. How many times does one step of the loop call `backward`?",
        options={
            "examples": "12: once for each example",
            "right": "1: once, on the loss of all 12 examples",
            "params": "25: once for each parameter, so that each one gets its own slope",
        },
        correct="right",
        why={
            "examples": "`mse` puts all 12 examples in one loss first. See the recipe in [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en).",
            "params": "One call gives every parameter its slope at once. See the recipe in [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en).",
            "right": "`mse` adds up all 12 examples in one loss, and one `backward(loss)` gives all 25 parameters their slope.",
        },
    )
    quick_backward
    return (quick_backward,)


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`m` has shape (2, 3), and `t = m.sum(axis=1)`. Why does `m / t` stop NumPy?",
        options={
            "right": "`t` has shape (2,), and lined up on the right, 3 and 2 differ",
            "sum": "NumPy cannot divide by a sum",
            "columns": "`t` has shape (3,): one total per column",
        },
        correct="right",
        why={
            "sum": "NumPy divides by any array whose shape fits. What is the shape of `t`? See the rule in [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "columns": "`axis=1` runs along each row, so there is one total per row. See [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "right": "One total per row gives shape (2,), and lined up on the right, (2, 3) and (2,) put 3 against 2. `keepdims=True` keeps `t` as a column, (2, 1), and then it fits.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your project in [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en) trains an MLP with 25 parameters, and every number in it is a `Value` of its own: each product is a Python object with a list of its parents. A small network that reads handwritten digits has about 100,000 parameters and learns from 60,000 images. One `Value` at a time, a single step over all those images would take hours. NumPy calculates with whole arrays at once, as in [Broadcasting](/course/open/np-broadcasting?lang=en), but it cannot give you slopes, and it only runs on the CPU, the computer's main processor. PyTorch can do both. Its arrays are called tensors: they calculate the way NumPy does, they can live on a GPU, a graphics processing unit, which does thousands of calculations at the same time, and, from the next lesson on, they remember where they came from, as your `Value` did.

    **An everyday example.** A dishwasher washes a whole load at once, much faster than washing every plate by hand. But it only washes what is inside it: a plate that is still on the table stays dirty.
    """)
    return


@app.cell(hide_code=True)
def predict_dtype():
    guess_dtype = L.guess(
        "dtype",
        question="`np.array([0.1, 0.2])` has dtype float64: every number takes 64 bits. What dtype do you think `torch.tensor([0.1, 0.2])` has?",
        options={
            "same": "float64, the same as NumPy",
            "right": "float32: 32 bits per number",
            "python": "No dtype: a tensor keeps Python's own floats",
        },
        reveal="**float32.** PyTorch keeps a float in 32 bits, half of NumPy's 64. That is about 7 digits instead of about 16: precise enough to train a network, in half the memory, and a GPU calculates much faster with it. So `torch.tensor(0.1).item()` gives `0.10000000149011612`: the float32 nearest to 0.1, written as a Python float. Whole numbers are int64 in both.",
    )
    guess_dtype
    return (guess_dtype,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        PyTorch is a library for neural networks, which you import with `import torch`. A **tensor** is a grid of numbers with any number of axes: one number has 0 axes, a vector 1, a matrix 2, and a batch of color photos 4 (photo, color, row, column). It is PyTorch's array. You make one from a list with `torch.tensor(...)`, as `np.array(...)` makes an array, and its `.shape`, `.dtype` and `.T` work as you know them from NumPy.

        A tensor calculates the way an array does: `+`, `-`, `*` and `/` number by number, `@` for matrix multiplication, and broadcasting with the same rule. Summaries such as `.sum()` and `.mean()` are methods of the tensor. Only the name of an axis differs: PyTorch calls it a `dim`. `t.mean(dim=1)` takes the mean along each row, and `keepdim=True` keeps that axis with length 1, as `keepdims=True` does in NumPy. PyTorch also understands NumPy's `axis=` and `keepdims=`, but its own documentation, and most code you will read, uses `dim` and `keepdim`. The mean squared error of a linear model therefore has the same formula, and the same lines of code, as with NumPy:
        """),
        L.formula(r"[[pred:\hat{\mathbf{y}}]] = \mathbf{X}[[param:\mathbf{w}]] + [[param:b]], \qquad [[loss:\mathcal{L}]] = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - [[pred:\hat{y}_i]]\right)^2"),
        mo.md(r"""
        In code: `y_pred = X @ w + b` and `loss = ((y - y_pred) ** 2).mean()`.

        **Float32 and int64.** A float that you put in a tensor becomes **float32** by default: 32 bits, about 7 digits. A tensor made from a NumPy array keeps NumPy's float64. A whole number is int64. PyTorch takes a mean of floats only, so the mean of a tensor of whole numbers stops it; `t.float()` gives a float32 copy. It does not mix the two in `@` either. So give data the dtype it needs when you make it: `torch.tensor(data, dtype=torch.float32)`.

        **A tensor with no axes.** `t.sum()` and `t.mean()` give back a tensor, not a number: `tensor(0.4375)`. That is on purpose: in the next lesson such a tensor remembers how it was made. `.item()` takes the number out, as a Python number, for a `print` or a list of losses.

        **Where a tensor lives.** Every tensor is stored on a **device**: the CPU, `"cpu"`, or a GPU: `"cuda"` on an NVIDIA card, `"mps"` on a Mac with an Apple chip. `t.device` says which device it is on. A GPU does thousands of calculations at the same time, so on a big tensor it is many times faster. One rule: the tensors of one calculation must be on the same device. PyTorch never moves a tensor by itself; it stops with an error such as "Expected all tensors to be on the same device". Until *Why AI uses GPUs, and the same code on yours*, a few lessons from now, your tensors stay on the CPU; that lesson shows how to move them.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **tensor** | | a grid of numbers with any number of axes: 0 for one number, 1 for a vector, 2 for a matrix. PyTorch's array | `torch.tensor([[1, 2], [3, 4]])` |
        | **float32** | float thirty-two | a float in 32 bits, about 7 digits: PyTorch's standard for floats, half the memory of NumPy's float64 | `dtype=torch.float32` |
        | `dim` | | PyTorch's name for an axis; `keepdim=True` keeps it with length 1 | `t.mean(dim=1, keepdim=True)` |
        | `.item()` | | the number in a tensor with no axes, as a Python number | `loss.item()` |
        | **device** | | where a tensor is stored and calculated: the CPU, or a GPU | `t.device` |
        """),
        L.watch_out("In physics and in pure maths, a tensor is something stricter: an object that changes in a fixed way when you change the coordinates. In machine learning, a tensor is just a grid of numbers with any number of axes. If you meet the word in a physics text, it means more there."),
        mo.md(r"""
        Three tensors of the worked example below, with 0, 1 and 2 axes. PyTorch writes a shape as `torch.Size`.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_tensors():
    draw.array_grids(C.picture(["0 axes", "1 axis", "2 axes"]),
                     title="Tensors with 0, 1 and 2 axes", box_width=66)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To calculate with PyTorch instead of NumPy:

    1. Make the data tensors, with floats: `torch.tensor(data, dtype=torch.float32)`.
    2. Calculate as with NumPy: `@`, broadcasting, and methods such as `.mean(dim=..., keepdim=True)`.
    3. For a single number, such as a loss, take it out with `.item()`.
    4. Check where your tensors live with `.device`: the tensors of one calculation must all be on the same device.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A linear model on 4 examples with 2 inputs each: its predictions and its mean squared error, with tensors. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_tensors():
    import torch

    # 1. the data as tensors of floats
    _X = torch.tensor([[1, 2], [3, 1], [0, 2], [2, 2]], dtype=torch.float32)   # (4, 2)
    _y = torch.tensor([5, 4, 3, 6], dtype=torch.float32)                       # (4,)
    _w = torch.tensor([1.0, 1.5])                                              # (2,)
    _b = 0.5
    # 2. calculate as with NumPy: (4, 2) @ (2,) + a number -> (4,)
    _y_pred = _X @ _w + _b
    _loss = ((_y - _y_pred) ** 2).mean()
    print(_y_pred)
    print(_loss)
    print(_X.mean(dim=0))   # the mean of each column: one per input
    print(_X.mean(dim=1, keepdim=True))   # the mean of each row, kept as a column: (4, 1)
    # 3. the loss as a Python number
    print(_loss.item())
    # 4. where the tensors live, and whether PyTorch finds a GPU from NVIDIA
    print(_X.device, _y_pred.device)
    print(torch.cuda.is_available())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `y_pred` is 4.5, 5, 3.5 and 5.5. The residuals $y - \hat{y}$ are 0.5, −1, −0.5 and 0.5, their squares 0.25, 1, 0.25 and 0.25, and their mean is 1.75 / 4 = 0.4375. `print(_loss)` shows `tensor(0.4375)`: still a tensor, with no axes. `.item()` gives the Python number `0.4375`. `_X.mean(dim=0)` runs down the rows and gives one mean per column: 1.5 and 1.75. `_X.mean(dim=1, keepdim=True)` runs along each row and gives one mean per row, 1.5, 2, 1 and 2, kept as a column of shape (4, 1), so that it fits `_X` for broadcasting. Both tensors are on the CPU, and the last line says whether PyTorch finds a GPU from NVIDIA on your computer. Either way, the next lessons run on the CPU.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_device():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two devices
    """)
    return


@app.cell(hide_code=True)
def choice_device_cell():
    choice_device = L.choice(
        "same-device",
        question="On a computer with an NVIDIA GPU, `X` lives on the GPU, `device(type='cuda', index=0)`, and `w` on the CPU. What does `X @ w` do?",
        options={
            "moves": "PyTorch moves `w` to the GPU by itself, because `X` is already there, and calculates on the GPU",
            "cpu": "PyTorch calculates on the CPU, where `w` is",
            "right": "PyTorch stops with an error: the two tensors are on different devices",
        },
        correct="right",
        why={
            "moves": "Moving a tensor takes time, so PyTorch leaves where it goes to you. See *Where a tensor lives* in *The idea*.",
            "cpu": "Then it would have to move `X`. Does PyTorch move a tensor by itself? See *Where a tensor lives* in *The idea*.",
            "right": "\"Expected all tensors to be on the same device\": the tensors of one calculation must be on the same device, and PyTorch never moves one by itself.",
        },
    )
    choice_device
    return (choice_device,)


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loss as a number

        `X` holds 3 examples with 2 inputs each, shape (3, 2), `y` their labels, shape (3,), and `w` and `b` are the weights and the bias term of a linear model. Replace the two `...`: `y_pred` the predictions for all examples at once, and `loss` their mean squared error, as a Python number. The check also tries other weights and bias terms.
        """),
        L.task("Make `loss` the mean squared error of `X @ w + b`, as a Python number."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("loss", lines=11)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "loss",
        [
            "Steps 2 and 3 of the recipe, as in the worked example.",
            "The predictions are a matrix times a vector, plus a number. The mean squared error stays a tensor with no axes until you take the number out.",
            "`y_pred = X @ w + b` and `loss = ((y - y_pred) ** 2).___().___()`.",
        ],
        why="$\\hat{\\mathbf{y}} = (2 + 2 - 1,\\ 0 + 6 - 1,\\ 1 + 2 - 1) = (3, 5, 2)$. The residuals $y - \\hat{y}$ are 1, −2 and 1, so the mean of their squares is (1 + 4 + 1) / 3 = 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_centered():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Each row minus its own mean

        `scores` holds the marks of two students in three tests, as whole numbers, with a row per student: shape (2, 3). Replace the `...`: `centered` is `scores` with each row minus that row's own mean, so you see for each student which test went better or worse than usual. The check also tries other scores.
        """),
        L.task("Make `centered` hold every score minus the mean of its own row."),
    ])
    return


@app.cell(hide_code=True)
def exercise_centered():
    box_centered = L.code_box("centered", lines=6)
    box_centered
    return (box_centered,)


@app.cell(hide_code=True)
def check_centered(box_centered):
    result_centered = L.check("centered", box_centered.value)
    result_centered
    return (result_centered,)


@app.cell(hide_code=True)
def hints_centered():
    L.hints(
        "centered",
        [
            "Steps 1 and 2 of the recipe: whole numbers first become floats, and then you take one mean per row.",
            "One mean per row is a summary along each row. To take it away from a matrix, keep it as a column, as `keepdims` did in [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "`centered = scores - scores.float().mean(dim=___, keepdim=___)`.",
        ],
        why="The rows have the means 7 and 6. (6, 8, 7) − 7 = (−1, 1, 0), and (4, 9, 5) − 6 = (−2, 3, −1).",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A tensor is the array you know from [Broadcasting](/course/open/np-broadcasting?lang=en): the same shapes, the same `@` and the same rule for stretching. What is new is small: float32, `dim`, `.item()` and the device. In [Training your MLP: the training loop](/course/open/nn-micrograd-train?lang=en) every number was a `Value` of its own; one tensor holds thousands of numbers in one object. In the next lesson, *PyTorch autograd does what your Value did*, a tensor learns to remember where it came from, and `backward()` gives every number in it its slope, as your own `Value` did.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_device, result_loss, result_centered):
    L.wrap_up(choice_device, result_loss, result_centered)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
