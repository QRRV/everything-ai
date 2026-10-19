# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-mlp-digits. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-mlp-digits", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en) and [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_eval():
    quick_eval = L.choice(
        "quick-eval",
        question="A network with dropout is switched to eval mode with `model.eval()`, and then predicts `model(X_test)`, without `torch.no_grad()`. What is the result?",
        options={
            "random": "Predictions that differ at every call, because dropout is still on",
            "error": "An error: in eval mode, PyTorch asks for `torch.no_grad()`",
            "right": "The right predictions, and a graph that nobody needs",
        },
        correct="right",
        why={
            "random": "`model.eval()` is the switch for dropout. See *The idea* in [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en).",
            "error": "The two switches are independent, and neither needs the other. See *The idea* in [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en).",
            "right": "Eval mode switches dropout off, so the predictions are right. Without `torch.no_grad()`, PyTorch still records a graph for them, which costs memory and time.",
        },
    )
    quick_eval
    return (quick_eval,)


@app.cell(hide_code=True)
def quick_check_relu():
    quick_relu = L.choice(
        "quick-relu",
        question="What does ReLU give for the inputs −2, 0.5 and 3?",
        options={
            "right": "0, 0.5 and 3",
            "abs": "2, 0.5 and 3",
            "step": "0, 1 and 1",
        },
        correct="right",
        why={
            "abs": "ReLU does not turn a negative number into a positive one. See [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).",
            "step": "That is a step. ReLU lets a positive number through as it is. See [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).",
            "right": "ReLU gives 0 for a negative number and lets a positive number through: max(0, z).",
        },
    )
    quick_relu
    return (quick_relu,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en) your project read 890 of 1,000 test digits, with one weighted sum of the pixels per digit. You worked out its slopes yourself. [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en) showed that a hidden layer lets a network learn more than one weighted sum can. And PyTorch now gives you all the parts: a module, an optimizer, a DataLoader and eval mode. Today you put them together into an MLP that reads the same digits, and you save it in your project. That is the last step of this part of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           7, here_label="← today", title="Where your project stands: your own neural network"),
        mo.md(r"""
        **An everyday example.** Someone who reads handwriting does not hold each digit against one fixed example. They look for parts, such as a loop at the top, a stroke straight down or a bar across, and put them together: a loop on a stroke is a 9, a loop on a loop an 8.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share-right",
        question="Your linear classifier read 890 of the 1,000 test digits right. An MLP with a hidden layer of 64 neurons trains on the same 6,000 digits, for 10 epochs. How many test digits do you think it reads right?",
        options={
            "same": "About 890: a hidden layer adds little",
            "right": "About 930",
            "all": "About 995: nearly all",
            "fewer": "Fewer than 890: with so many weights it guesses",
        },
        reveal="**About 930.** In the demo at the end of this lesson, the MLP reads 929 of the 1,000 test digits right. It misreads 71 of them, where the linear classifier misread 110: about a third fewer mistakes. Not all: some digits are written so that even people hesitate, and 6,000 examples are not many.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An MLP for the digits has two layers. The **hidden layer**, `nn.Linear(784, 64)` with ReLU, gives every image 64 numbers: each hidden neuron has its own weighted sum of the 784 pixels, its own pattern, as in [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en). The **output layer**, `nn.Linear(64, 10)`, makes the 10 logits from those 64 numbers, one logit per digit. So a digit no longer has one fixed pattern: its logit combines 64 of them. That takes more numbers: 784 · 64 + 64 = 50,240 for the hidden layer and 64 · 10 + 10 = 650 for the output layer, 50,890 in all, against 7,850 for your linear classifier.

        The loss is the cross-entropy of [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en), as the mean over the $n$ examples of a mini-batch:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\frac{1}{n} \sum_{i=1}^{n} \log([[pred:\hat{y}_{i,c_i}]])"),
        mo.md(r"""
        Here $c_i$ is the right digit of example $i$, and $\hat{y}_{i,c_i}$ the probability that softmax gives that digit. In PyTorch it is **`nn.CrossEntropyLoss()`**. It gets the logits, shape $(n, 10)$, and the labels as digits, shape $(n,)$, as int64, and it does the softmax itself. So the model gives back logits, without softmax. The predicted digit is the one with the largest logit, `logits.argmax(dim=1)`, as with `argmax(axis=1)` in NumPy: softmax does not change which logit is largest.

        | | Your linear classifier | Your MLP in PyTorch |
        |---|---|---|
        | the model | `X @ W.T + b` | `DigitsMLP`: `nn.Linear(784, 64)`, ReLU, `nn.Linear(64, 10)` |
        | numbers it learns | 7,850 | 50,890 |
        | the loss | cross-entropy, written out with NumPy | `nn.CrossEntropyLoss()` on the logits |
        | the slopes | your own formula, `diff.T @ X / n` | `loss.backward()` |
        | the step | `W = W - lr * grad_W` | `opt.step()` |
        | the data per step | all 6,000 examples | a mini-batch of 64 from a DataLoader |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | `nn.CrossEntropyLoss()` | n n cross-entropy loss | the cross-entropy as a PyTorch module: softmax and the mean of minus the log of the right probability, in one | `loss_fn = nn.CrossEntropyLoss()` |
        | $c_i$ | c i | the right class of example $i$: here its digit | `y[i]` |
        | $\hat{y}_{i,c_i}$ | y hat i c i | the probability the model gives the right digit of example $i$ | |
        """),
        L.watch_out("`nn.CrossEntropyLoss` does the softmax itself. A model that ends with softmax has it done twice: the loss still goes down, but slowly, and the model learns less. Give the loss the logits, and the labels as whole numbers, int64, not as floats."),
        mo.md(r"""
        The picture: a mini-batch of $n$ images through the MLP, with the shape of what each layer gives.
        """),
        draw.flow([("Mini-batch", ["n images", "of 784 pixels", "(n, 784)"]), ("Hidden layer", ["64 neurons", "with ReLU", "(n, 64)"]),
                   ("Output layer", ["10 logits", "per image", "(n, 10)"])],
                  ["nn.Linear(784, 64)", "nn.Linear(64, 10)"], ["", ""], title="A mini-batch of n images through DigitsMLP"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train a classifier for the digits in PyTorch:

    1. Load the digits as tensors: the pixels as float32 divided by 255, the labels as int64. Put the training digits in a DataLoader that shuffles.
    2. Make the model: it gives back 10 logits per image, without softmax.
    3. Make the loss and the optimizer: `loss_fn = nn.CrossEntropyLoss()` and `opt = torch.optim.SGD(model.parameters(), lr=...)`.
    4. Train for a few epochs. For each mini-batch: `loss = loss_fn(model(xb), yb)`, then `opt.zero_grad()`, `loss.backward()` and `opt.step()`.
    5. Measure on the test digits, in eval mode and without a graph: the predictions are `model(X_test).argmax(dim=1)`, and the accuracy is the share that is right, `(pred == y_test).float().mean()`: `.float()` turns True and False into 1 and 0, because PyTorch takes a mean of floats only.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    First the recipe with the model you know: your linear classifier, now as one layer `nn.Linear(784, 10)` that gives 10 logits per image. The cell trains it for 3 epochs of mini-batches of 64, with $\eta$ = 0.5, and prints the loss over all training digits before training and after each epoch. Then it measures the model on the 1,000 test digits. The comments with `#` say which step of the recipe each part is. Run the cell; it takes a second.
    """)
    return


@app.cell
def example_linear():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. the digits as tensors, and a DataLoader that shuffles
    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(data["test_labels"], dtype=torch.int64)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    # 2. the model: one layer from 784 pixels to 10 logits
    torch.manual_seed(0)
    model = nn.Linear(784, 10)

    # 3. the loss and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    print(loss_fn(model(X), y).item())

    # 4. three epochs of mini-batches
    torch.manual_seed(2)
    for epoch in range(3):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(epoch, loss_fn(model(X), y).item())

    # 5. the accuracy on the test digits, in eval mode and without a graph
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
    model.train()
    print(pred[:10])
    print(y_test[:10])
    print((pred == y_test).float().mean().item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Before training, the loss is about 2.35, close to 2.30: the loss of a probability of 0.1 for the right digit, because small random weights give ten logits close to each other. After 3 epochs it is about 0.28. Of the first ten test digits, the model reads nine right: the ninth is a 5, and the model says 6. Of all 1,000, it reads 887 right, an accuracy of 0.887: about what your own linear classifier read, 890 after 100 steps over all the data. It is the same kind of model; what changes it is a hidden layer, which you add in exercise 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_input():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What the loss gets
    """)
    return


@app.cell(hide_code=True)
def choice_input_cell():
    choice_input = L.choice(
        "loss-input",
        question="`loss_fn = nn.CrossEntropyLoss()`, and `xb` is a mini-batch of 64 images. What do you give `loss_fn`?",
        options={
            "probs": "The probabilities after softmax, shape (64, 10), and the right digits, shape (64,)",
            "right": "The logits, shape (64, 10), and the right digits, shape (64,)",
            "argmax": "The predicted digits and the right digits, both of shape (64,)",
        },
        correct="right",
        why={
            "probs": "`nn.CrossEntropyLoss` does the softmax itself; with probabilities, it would do it twice. See *The idea*.",
            "argmax": "A predicted digit is a whole number: a small change in a weight hardly ever changes it, so it gives no slopes to learn from. See *The idea*.",
            "right": "The logits of the model, one row per image, and the right digit of each image as a whole number: `nn.CrossEntropyLoss` does the softmax and takes minus the log of the right probability.",
        },
    )
    choice_input
    return (choice_input,)


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your MLP as a module

        `DigitsMLP(hidden)` is an MLP for the digits with `hidden` neurons in its hidden layer. `__init__` already makes the hidden layer: `hidden` neurons, each with 784 weights. Replace the two `...`: the output layer, from `hidden` numbers to 10 logits, and in `forward` the mini-batch `x` through the hidden layer, ReLU and the output layer. The check makes `DigitsMLP(64)` and `DigitsMLP(16)`, each after `torch.manual_seed(0)`, gives them three test digits and compares the logits with those of the course's model. When your class is right, the check saves it in your project, as `work/projects/nn/torch_mlp.py`.

        That file starts with `from torch import nn`: that `nn` is PyTorch's, even though your project is called `nn` too. You import a file of your project by its full name, such as `from nn.torch_mlp import DigitsMLP`.
        """),
        L.task("Give `DigitsMLP` its output layer, and a `forward` that gives back 10 logits per image."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("digits-mlp", lines=14)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("digits-mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "digits-mlp",
        [
            "Step 2 of the recipe: the two layers of the picture, with ReLU in between.",
            "The output layer gets the `hidden` numbers of the hidden layer and gives 10. `forward` works from the inside out: first the hidden layer on `x`, then ReLU, then the output layer.",
            "`self.out = nn.Linear(___, 10)`, and in `forward`: `return self.out(torch.relu(self.___(x)))`.",
        ],
        why="`self.out = nn.Linear(hidden, 10)` turns the `hidden` numbers of each image into 10 logits. `self.hidden(x)` gives each image `hidden` weighted sums, `torch.relu` sets the negative ones to 0, and `self.out` makes the logits. No softmax: `nn.CrossEntropyLoss` does that.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Train

        `fit(model, loader, epochs, lr)` trains a model on the mini-batches of a DataLoader, with `nn.CrossEntropyLoss` and SGD, and gives back the mean loss of each epoch. The loss, the optimizer, the loops and the sum of the losses are there. Replace the `...` with step 4 of the recipe for one mini-batch, and then add the loss of the mini-batch to `total`, as a number: the line after the loop divides that sum by the number of mini-batches. The check trains a small `nn.Linear(2, 3)` on four examples, with `fit(model, loader, 3, 0.5)`, and also with another batch size, number of epochs and learning rate. When your function is right, the check saves it in your project, as `work/projects/nn/torch_train.py`.
        """),
        L.task("Make `fit` take an update step for every mini-batch, with the cross-entropy of its logits, and add up the losses of the epoch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-torch", lines=16)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit-torch", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-torch",
        [
            "Inside the loop over the mini-batches: step 4 of the recipe, and then keep count of the loss for the mean of the epoch.",
            "First the loss of this mini-batch, from its logits and its labels. Then the three lines of the optimizer: clear the slopes, work them out, step.",
            "`loss = loss_fn(model(___), ___)`, then `opt.zero_grad()`, `loss.___()` and `opt.step()`, and last `total = total + loss.___()`.",
        ],
        why="`loss_fn(model(xb), yb)` is the cross-entropy of this mini-batch. `opt.zero_grad()` clears the slopes of the step before, `loss.backward()` works out the new ones, and `opt.step()` moves every parameter against its slope. `total = total + loss.item()` adds the loss of the mini-batch as a number, for the mean of the epoch. On the four examples, the mean loss goes from about 1.11 in the first epoch to 0.89 in the third.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below makes your own `DigitsMLP(64)`, trains it with your own `fit` on the 6,000 training digits, for 10 epochs of mini-batches of 64 with $\eta$ = 0.5, and measures it on the 1,000 test digits, which it never saw. That takes a second or two. The cell uses your own modules, so it works once exercises 2 and 3 are done.
    """)
    return


@app.cell(hide_code=True)
def demo_digits(result_fit, result_mlp):
    result_mlp, result_fit  # run again when exercise 2 or 3 saves your module
    _result = C.trained()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `DigitsMLP` and your `fit` are in your project, and this cell uses them.*")
    elif _result == "slow":
        _out = mo.md("*Training with your `fit` in `work/projects/nn/torch_train.py` takes more than a minute on the 6,000 digits. Does it go once over the mini-batches of `loader` in each epoch?*")
    elif _result == "mlp":
        _out = mo.md("*The `DigitsMLP` in `work/projects/nn/torch_mlp.py` no longer gives the logits the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your class again.*")
    elif _result == "fit":
        _out = mo.md("*The `fit` in `work/projects/nn/torch_train.py` no longer trains the way the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*")
    else:
        _accuracy, _losses = _result
        _digit, _baseline = C.baseline()
        _right = round(_accuracy * 1000)
        _out = mo.vstack([
            mo.md(f"""Your MLP reads <span class="ea-num">{L.number(_right, 0)}</span> of the 1,000 test digits right: an accuracy of <span class="ea-num">{L.number(_accuracy, 3)}</span>. Your linear classifier read {L.number(C.LINEAR, 0)}, and always saying the most common digit, a {_digit}, gets <span class="ea-num">{L.number(_baseline, 3)}</span>. So your MLP misreads {L.number(1000 - _right, 0)} digits, where the linear classifier misread {L.number(1000 - C.LINEAR, 0)}."""),
            mo.Html(C.losses_picture(_losses, {"title": "The loss of your training, per epoch", "epoch": "epoch", "loss": "mean loss",
                                               "alt": f"A line through 10 points, one per epoch: the mean loss of the mini-batches falls from {L.number(_losses[0], 2)} in the first epoch to {L.number(_losses[-1], 2)} in the tenth, fast at first and then slower."})),
            mo.md(f"""Each point is the mean loss of the 94 mini-batches of one epoch, as your `fit` gives it back: from <span class="c-loss ea-num">{L.number(_losses[0], 2)}</span> in the first epoch to <span class="c-loss ea-num">{L.number(_losses[-1], 2)}</span> in the tenth. On the training digits, the loss keeps going down. Whether that also holds for digits the model never saw, you find out in *Reading loss curves*, two lessons on."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The pieces of the PyTorch lessons fit together here: a module as in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en), an optimizer, a DataLoader that shuffles, and eval mode from [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en) to measure. With the hidden layer of [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en), your MLP makes about a third fewer mistakes than the linear classifier of [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en). Your project `nn` now has `torch_mlp.py` and `torch_train.py` as well. In the next lesson, *Why AI uses GPUs, and the same code on yours*, you count the work your MLP does, and let the same training loop use a GPU when there is one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_input, result_fit, result_mlp):
    L.wrap_up(choice_input, result_mlp, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
