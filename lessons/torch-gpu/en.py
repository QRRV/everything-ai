# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-gpu. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-gpu", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) and [PyTorch tensors](/course/open/torch-tensors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="How many weights does `nn.Linear(784, 64)` have, without its bias terms?",
        options={
            "right": "50,176: 784 times 64",
            "sum": "848: 784 + 64",
            "in": "784: one per pixel",
        },
        correct="right",
        why={
            "sum": "Each of the 64 neurons has its own weight for each of the 784 inputs. See *The idea* in [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "in": "Those are the weights of one neuron; the layer has 64 neurons. See *The idea* in [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "right": "784 · 64 = 50,176: one weight for every pair of an input and a neuron. With the 64 bias terms, the layer has 50,240 parameters.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="`t = torch.tensor([1.0, 2.0])`. On which device is `t`?",
        options={
            "gpu": "On the GPU, if the computer has one",
            "none": "On none yet: only a calculation puts it on a device",
            "right": "On the CPU, until you move it",
        },
        correct="right",
        why={
            "gpu": "PyTorch makes a new tensor on the CPU, even when there is a GPU, and never moves it by itself. See *Where a tensor lives* in [PyTorch tensors](/course/open/torch-tensors?lang=en).",
            "none": "Every tensor is stored somewhere from the moment it is made. See *Where a tensor lives* in [PyTorch tensors](/course/open/torch-tensors?lang=en).",
            "right": "A new tensor is made on the CPU, unless you say otherwise; `t.device` says `cpu`.",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your MLP for the digits trained in a second or two. The models behind chatbots and image generators have from about a billion to a trillion parameters, against 50,890 in yours, and train on far more data. They are trained on GPUs, chips that were made for graphics, because the work of a neural network is the same simple calculation done an enormous number of times. To know what a model costs, you count that work. And to use a GPU from PyTorch, you need only a few lines, which also run on a computer without one.

    **An everyday example.** A restaurant has a few experienced cooks, who can make any dish. For a wedding, the kitchen also needs a mountain of vegetables cut: then a hundred helpers who can only cut, all at the same time, are done long before the cooks.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="On a computer with a GPU, you train the small `DigitsMLP(64)` of the project lesson on mini-batches of 64 digits, once on the CPU and once on the GPU. What do you think happens?",
        options={
            "much": "The GPU is much faster: it has thousands of cores",
            "right": "About as fast, and sometimes even slower than the CPU",
            "never": "It does not run: a GPU only takes big networks",
        },
        reveal="**About as fast, sometimes even slower.** A mini-batch of 64 digits costs this network about 20 million FLOPs, too little to keep thousands of cores busy. And every mini-batch must first travel to the GPU, and every `.item()` back. A GPU pays off when the matrix products are big: wide layers, large mini-batches, large models. Then it is many times faster.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Counting the work.** As in [Too many weights to nudge one by one](/course/open/nn-why-backprop?lang=en), every weight costs 2 FLOPs per example in a forward pass: one multiplication and one addition. `nn.Linear(n_in, n_out)` has `n_in * n_out` weights, so `DigitsMLP(64)`, with 784 · 64 + 64 · 10 = 50,816 weights, costs 101,632 FLOPs to read one image. A training step costs about three times as much, 6 FLOPs per weight per example, because the backward pass uses every weight twice: once to work out its slope and once to pass the slopes back to the layer before. An epoch of 6,000 images then costs about 6 · 6,000 · 50,816 = 1,829,376,000 FLOPs, close to 2 billion.

        **Why a GPU.** Those FLOPs are independent of each other: every number a layer gives back can be worked out on its own. A CPU has a few fast cores that can do anything, from 4 to about 24 in a laptop. A **GPU** has thousands of simple cores that all do the same calculation, each on other numbers, at the same time. For the large matrix products of a big network, that is many times faster. For a small network it can even be slower, because moving the numbers to the GPU also takes time.

        **The same code on every computer.** As in [PyTorch tensors](/course/open/torch-tensors?lang=en), the tensors of one calculation must be on the same device, and PyTorch never moves them by itself. So:

        - choose the device once, with an `if`: `"cuda"`, an NVIDIA GPU, when `torch.cuda.is_available()` is `True`, and `"cpu"` otherwise; on a Mac with an Apple chip, the test is `torch.backends.mps.is_available()` and the device `"mps"`;
        - move the model, once, before you make the optimizer: `model.to(device)` moves all its parameters;
        - move every mini-batch: `xb.to(device)` and `yb.to(device)`;
        - bring a result back as a Python number with `.item()`.

        On a computer without a GPU, `device` is `"cpu"`, and the code runs as it always did.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **GPU** | G P U | a chip with thousands of simple cores that do the same calculation on many numbers at once | `"cuda"`, `"mps"` |
        | `.to(device)` | to device | moves a model or a tensor to a device; for a tensor it gives a moved copy back | `model.to(device)`, `xb.to(device)` |
        | `torch.cuda.is_available()` | | `True` when PyTorch can use an NVIDIA GPU | `if torch.cuda.is_available():` |
        """),
        L.watch_out("On a computer without a GPU, a forgotten `.to(device)` gives no error: everything is on the CPU anyway. The error only comes on a computer with a GPU, maybe someone else's. That is why the check of exercise 2 also tries your code on PyTorch's `\"meta\"` device, a stand-in for a GPU that only knows the shapes of tensors, not their numbers, and stops just like a GPU when a tensor is on another device."),
        mo.md(r"""
        The picture: what lives where while a network trains on a GPU.
        """),
        draw.flow([("CPU", ["your program", "the DataLoader", "plain numbers"]), ("GPU", ["the model", "each mini-batch", "the loss"])],
                  [".to(device)"], [".item()"], title="Training on a GPU"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To let a training loop use a GPU when there is one:

    1. Choose the device once, with an `if` and an `else`: `device = "cuda"` when `torch.cuda.is_available()`, else `device = "cpu"`.
    2. Move the model with `model.to(device)`, and only then make the optimizer.
    3. In the loop, move every mini-batch: `xb, yb = xb.to(device), yb.to(device)`.
    4. Bring results back as numbers with `.item()`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The MLP of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), trained for one epoch with the recipe. `weights` counts the weights of its two `nn.Linear` layers. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_device():
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
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    # 1. choose the device once
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    torch.manual_seed(0)
    model = DigitsMLP(64)
    weights = 784 * 64 + 64 * 10
    print(2 * weights, 6 * 6000 * weights)
    # 2. move the model, then make the optimizer
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    print(model.out.weight.device)

    torch.manual_seed(2)
    total = 0
    for xb, yb in loader:
        # 3. move every mini-batch
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 4. back as a Python number
        total = total + loss.item()
    print(total / len(loader))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    On a computer without a GPU, the cell prints `cpu`, and `model.out.weight` is on the CPU too: nothing moved, and nothing had to. On a Mac with an Apple chip it prints `cpu` as well, because this code only looks for an NVIDIA GPU. On a computer with an NVIDIA GPU it prints `cuda`, and the weights are on `cuda:0`, the first GPU. Either way, the mean loss of the epoch is about 0.76, the first point in the project lesson's demo. The forward pass of one image costs 101,632 FLOPs, and an epoch of training about 1.8 billion: less than a second of work for a CPU. A network with a billion parameters needs 2 billion FLOPs for a single input.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nogpu():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. No GPU
    """)
    return


@app.cell(hide_code=True)
def choice_nogpu_cell():
    choice_nogpu = L.choice(
        "no-gpu",
        question="Your laptop has no GPU. You run the training loop of the worked example. What happens?",
        options={
            "right": "It runs on the CPU, as before",
            "error": "It stops with an error: there is no CUDA",
            "slower": "It runs, but slower than the same loop without the device lines",
        },
        correct="right",
        why={
            "error": "`torch.cuda.is_available()` is then `False`, so `device` becomes `\"cpu\"`. See *The same code on every computer* in *The idea*.",
            "slower": "Moving to the device you are already on costs nothing. See *The same code on every computer* in *The idea*.",
            "right": "`device` becomes `\"cpu\"`, and `.to(\"cpu\")` leaves model and data where they are: the code runs as before.",
        },
    )
    choice_nogpu
    return (choice_nogpu,)


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Predict on a device

        `predict(model, X, device)` gives the predicted digit for every image in `X`, worked out on `device`. Replace the two `...`: first move the model to `device`; then give it `X`, moved to `device` as well. The check makes your `DigitsMLP(16)` and calls `predict` on five test digits, on the CPU and on the `"meta"` device of the watch-out. On `"meta"` it shows where the predictions are, their shape, and where the model is.
        """),
        L.course_versions(),
        L.task("Move the model and the images to `device` before the model predicts."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict-device", lines=12)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict-device", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict-device",
        [
            "Steps 2 and 3 of the recipe, for a prediction instead of a training step.",
            "The model moves with one line before it predicts. The images move in the line where the model gets them.",
            "`model.___(device)` on the first `...`, and `model(X.___(device))` on the second.",
        ],
        why="`model.to(device)` moves all parameters of the model; `X.to(device)` gives a copy of the images on that device. Then the model and its input are on the same device, and so are the predictions. On `\"meta\"` there are no numbers, only the shape: five predictions.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_moves():
    mo.md(r"""
    ### Exercise 3. What moves?

    Five things from a training loop on a GPU. Put each in its group: moves to the device, or needs no move.
    """)
    return


@app.cell(hide_code=True)
def exercise_moves():
    sort_moves = L.sort("what-moves")
    sort_moves
    return (sort_moves,)


@app.cell(hide_code=True)
def check_moves(sort_moves):
    result_moves = L.check("what-moves", sort_moves)
    result_moves
    return (result_moves,)


@app.cell(hide_code=True)
def hints_moves():
    L.hints(
        "what-moves",
        [
            "For each thing, ask: is it, or does it hold, a tensor that the GPU calculates with?",
            "Two things move, three stay. Plain Python numbers never need to move.",
            "The model and every mini-batch move. Is a learning rate a tensor? And what does `.item()` give back?",
        ],
        why="**Moves to the device:** the model, with its parameters, and every mini-batch. **Needs no move:** the learning rate and the number of epochs, plain Python numbers, and `loss.item()`, which is already a Python number back on the CPU.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The devices of [PyTorch tensors](/course/open/torch-tensors?lang=en) now run a whole training loop: the model goes to the device once, every mini-batch each time. Counting FLOPs, about 6 per weight per example in training, tells you what the networks of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) and far bigger ones cost. In the next lesson, *Reading loss curves*, you look at what training does epoch by epoch, on the training data and on data the model never saw.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_nogpu, result_moves, result_predict):
    L.wrap_up(choice_nogpu, result_predict, result_moves)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
