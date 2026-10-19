# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-batchnorm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-batchnorm", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) and [Looking inside a training run](/course/open/nn-activation-stats?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_standardized():
    quick_standardized = L.choice(
        "quick-standardized",
        question="You standardize a column of numbers: minus its mean, divided by its standard deviation. What are its mean and standard deviation afterwards?",
        options={
            "swapped": "Mean 1 and standard deviation 0",
            "right": "Mean 0 and standard deviation 1",
            "kept": "Mean 0, and the standard deviation it had before",
        },
        correct="right",
        why={
            "swapped": "Taking off the mean leaves a mean of 0; dividing by the standard deviation makes it 1. See [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "kept": "Dividing by the standard deviation changes it too: afterwards it is 1. See [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en).",
            "right": "Taking off the mean gives mean 0, and dividing by the standard deviation gives standard deviation 1. Today a layer does that inside the network.",
        },
    )
    quick_standardized
    return (quick_standardized,)


@app.cell(hide_code=True)
def quick_check_dead_layer():
    quick_dead_layer = L.choice(
        "quick-dead-layer",
        question="Every neuron of one hidden layer of an MLP is dead. What do the layers before it get on the way back?",
        options={
            "same": "The same gradient as before: a dead layer passes it on",
            "right": "A gradient of 0: nothing comes back through the dead layer",
            "larger": "A larger gradient, to make up for the dead layer",
        },
        correct="right",
        why={
            "same": "A dead neuron has a slope of 0 for every example, so it multiplies whatever comes back by 0. See [Looking inside a training run](/course/open/nn-activation-stats?lang=en).",
            "larger": "Nothing makes up for it: every gradient through the dead layer is multiplied by a slope of 0. See [Looking inside a training run](/course/open/nn-activation-stats?lang=en).",
            "right": "Every path back runs through the dead layer, where the slope of ReLU is 0. The layers before it stop learning too.",
        },
    )
    quick_dead_layer
    return (quick_dead_layer,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Looking inside a training run](/course/open/nn-activation-stats?lang=en), a learning rate that was far too large killed a whole layer. He starting weights from [Initialization](/course/open/nn-initialization?lang=en) give every layer a good start, but training moves the weights, and with them the mean and spread of the $z$'s that each layer gives. The deeper the network, the more layers can drift. Gradient descent likes features with mean 0 and standard deviation 1, as you saw in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en). Batch normalization gives every layer exactly that, at every step of training.
    """)
    return


@app.cell(hide_code=True)
def predict_deep():
    guess_deep = L.guess(
        "deep-with-bn",
        question="An MLP with 10 hidden layers of 128 neurons and He starting weights trains for one epoch on the 6,000 training digits, with a learning rate of 0.3. It reads 16% of the 1,000 test digits right, hardly better than guessing. Now batch normalization goes between every hidden layer and its ReLU; everything else stays the same. How many test digits do you think it reads right now?",
        options={
            "same": "About the same: 16%",
            "some": "Somewhat more: about 40%",
            "right": "Most of them: about 90%",
        },
        reveal="**About 90%.** Without batch normalization, the loss stays at about 2.3 for the whole epoch; with it, the loss falls to about 0.43, and the network reads 888 of the 1,000 test digits right. Same layers, same starting weights, same learning rate: only the standardizing inside differs.",
    )
    guess_deep
    return (guess_deep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Batch normalization**, BatchNorm for short, does to the $z$'s of a layer what standardizing did to features: for each neuron, it takes off the mean and divides by the standard deviation, over the examples in the mini-batch. For the $z$'s $z_1, \dots, z_m$ of one neuron in a mini-batch of $m$ examples:
        """),
        L.formula(r"\mu_B = \frac{1}{m} \sum_{i=1}^{m} z_i"),
        L.formula(r"\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (z_i - \mu_B)^2"),
        L.formula(r"\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + [[hyper:\varepsilon]]}} \qquad y_i = [[param:\gamma]] \, \hat{z}_i + [[param:\beta]]"),
        mo.md(r"""
        Then it rescales: $\gamma$ and $\beta$ are learned, one of each per neuron, so that the network can give a neuron another mean and spread when that helps; with $\gamma = \sigma_B$ and $\beta = \mu_B$ it could even undo the standardizing. They start at 1 and 0. $\varepsilon$, usually 0.00001, keeps it from dividing by 0. In PyTorch, `nn.BatchNorm1d(128)` does this for a layer of 128 neurons; it goes between the `nn.Linear` and the ReLU, and `bn.weight` is its $\gamma$, `bn.bias` its $\beta$.

        In a mini-batch, every answer depends on the other examples in it. After training, you want one answer per example, whatever batch it is in. So during training, BatchNorm also keeps **running averages** of the mean and variance of each neuron: after every batch, 0.9 times the old running average plus 0.1 times the batch's own. For the variance, it divides by $m - 1$ there instead of by $m$, a correction for estimating the variance of all data from a sample. In eval mode, `model.eval()`, it uses those instead of the batch's own; `model.train()` switches back.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **batch normalization** | | for each neuron: standardize its $z$'s over the mini-batch, then times $\gamma$ plus $\beta$ | `nn.BatchNorm1d(128)` |
        | $\mu_B$, $\sigma_B^2$ | mu B, sigma B squared | the mean and the variance of a neuron's $z$'s over the mini-batch $B$, dividing by $m$ | `Z.mean(dim=0)`, `((Z - mean) ** 2).mean(dim=0)` |
        | $\hat{z}$ | z hat | a standardized $z$: here the hat means standardized, not estimated | `(Z - mean) / torch.sqrt(var + 1e-5)` |
        | $\gamma$, $\beta$ | gamma, beta | the learned scale and bias term of each neuron after standardizing; they start at 1 and 0 | `bn.weight`, `bn.bias` |
        | $\varepsilon$ | epsilon | a small number that keeps you from dividing by 0 | `eps=1e-5` |
        | **running averages** | | the mean and variance per neuron that BatchNorm keeps during training, for eval mode | `bn.running_mean`, `bn.running_var` |
        """),
        L.watch_out("Call `model.eval()` before you test or use a network with BatchNorm, and `model.train()` before you train it again. In train mode, every batch is standardized with its own mean and variance, and the running averages change with whatever you give it. And keep batches large enough: the mean and variance of 4 examples are rough guesses. Why BatchNorm helps so much is still discussed: the original paper said that each layer's inputs keep shifting during training; later work points to a smoother loss surface. That it helps is clear."),
        mo.md(r"""
        The picture: the training loss of the network of the guess during its one epoch, the mean of the last 10 mini-batches. Without BatchNorm, it hardly moves from 2.30, the loss of guessing. With BatchNorm it falls to about 0.43. The same holds for other learning rates: without BatchNorm, only 0.1 trains well, to 66% of the test digits right; at 0.3 the network is stuck and at 1.0 the loss becomes `nan`. With BatchNorm, 0.1, 0.3 and 1.0 all reach 84% to 89%.
        """),
        mo.Html(C.loss_picture({"title": "10 hidden layers, learning rate 0.3, one epoch", "x": "mini-batch", "y": "training loss",
                                "plain": "without BatchNorm", "normed": "with BatchNorm",
                                "alt": "Two loss curves over 94 mini-batches. Without BatchNorm, dashed, the loss stays between about 2.1 and 2.4. With BatchNorm, it falls from about 2.3 to about 0.4."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Batch normalization of a batch of $z$'s, one row per example and one column per neuron:

    1. Per column: the mean and the variance over the batch, dividing by the number of rows.
    2. Standardize: minus the mean, divided by the square root of the variance plus $\varepsilon$.
    3. Rescale: times $\gamma$, plus $\beta$.
    4. In train mode, update the running averages: 0.9 times the old plus 0.1 times the batch's own mean and variance, the variance this time divided by the number of rows minus 1. In eval mode, use them in step 2 instead of the batch's own.

    In a model, it goes Linear, then BatchNorm, then ReLU, with one `nn.BatchNorm1d` for every hidden layer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A batch of 4 examples, the $z$'s of a layer with 2 neurons: the first between 1 and 6, the second between 40 and 70. `torch.sqrt` takes the square root of every number in a tensor, as `** 0.5` does, and `torch.ones(2)` and `torch.zeros(2)` make a tensor of two 1s and one of two 0s. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_batchnorm():
    import torch
    from torch import nn

    _Z = torch.tensor([[1.0, 50.0], [2.0, 70.0], [3.0, 40.0], [6.0, 60.0]])
    # 1. per column: the mean and the variance over the batch, dividing by the number of rows
    _mean = _Z.mean(dim=0)
    _var = ((_Z - _mean) ** 2).mean(dim=0)
    print(_mean, _var)
    # 2. standardize, with eps = 0.00001
    _Z_hat = (_Z - _mean) / torch.sqrt(_var + 1e-5)
    # 3. rescale with gamma and beta, which start at 1 and 0
    _gamma = torch.ones(2)
    _beta = torch.zeros(2)
    print(_gamma * _Z_hat + _beta)
    # 4. nn.BatchNorm1d in train mode does steps 1 to 3, and updates its running averages
    _bn = nn.BatchNorm1d(2)
    with torch.no_grad():
        print(_bn(_Z))
    # then: also print _bn.running_mean, and compare it with 0.1 * _mean; run again
    # and then: switch _bn to eval mode with _bn.eval(), print _bn(_Z) once more, and run again
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The means are 3 and 55, the variances 3.5 and 125. After standardizing, both columns have mean 0 and standard deviation 1: the first neuron's $z$'s become −1.07, −0.53, 0 and 1.60, the second's −0.45, 1.34, −1.34 and 0.45, whatever their scale was. With $\gamma = 1$ and $\beta = 0$, rescaling changes nothing yet. `nn.BatchNorm1d(2)` gives the same numbers. Its running means were 0 at the start; after this one batch they are $0.9 \cdot 0 + 0.1 \cdot 3 = 0.3$ and $0.1 \cdot 55 = 5.5$: far from 3 and 55 still. Only after many batches are they close to the mean of all the data. So in eval mode, after `_bn.eval()`, this one batch would come out far from standardized: its first column as 0.60, 1.45, 2.31 and 4.88.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_eval():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Testing without eval mode
    """)
    return


@app.cell(hide_code=True)
def choice_eval_cell():
    choice_eval = L.choice(
        "eval-mode",
        question="A network with BatchNorm has been trained. You test it on the test digits in batches of 10, but you forget `model.eval()`. What happens?",
        options={
            "nothing": "Nothing changes: BatchNorm only matters while the network learns",
            "frozen": "BatchNorm uses its running averages, but stops updating them",
            "right": "Each answer then depends on the other 9 digits in its batch",
        },
        correct="right",
        why={
            "nothing": "BatchNorm does not know whether you are learning or testing; it only knows its mode. In train mode it standardizes every batch with that batch's own mean and variance.",
            "frozen": "That is what eval mode does. Without `model.eval()`, BatchNorm stays in train mode: it uses each batch's own mean and variance, and even updates its running averages with your test digits.",
            "right": "In train mode, each batch of 10 is standardized with its own mean and variance, so the answer for one digit depends on which 9 others come with it, and the running averages change with your test digits too. `model.eval()` before testing, `model.train()` before training again.",
        },
    )
    choice_eval
    return (choice_eval,)


@app.cell(hide_code=True)
def text_exercise_bn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Batch normalization by hand

        Write `batch_norm(Z, gamma, beta)`: batch normalization in train mode, by steps 1 to 3 of the recipe, with $\varepsilon$ = 0.00001. `Z` has one row per example and one column per neuron; `gamma` and `beta` have one number per column. The check calls `batch_norm(torch.tensor([[1.0, 10.0], [2.0, 20.0], [6.0, 30.0]]), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]))`, which should give `[[-0.9258, -1.4495], [-0.4629, 1.0], [1.3887, 3.4495]]`, and tries other batches.
        """),
        L.task("Make `batch_norm` standardize each column over the batch, then times gamma, plus beta."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bn():
    box_bn = L.code_box("batch-norm", lines=6)
    box_bn
    return (box_bn,)


@app.cell(hide_code=True)
def check_bn(box_bn):
    result_bn = L.check("batch-norm", box_bn.value)
    result_bn
    return (result_bn,)


@app.cell(hide_code=True)
def hints_bn():
    L.hints(
        "batch-norm",
        [
            "Step 1 as in the worked example: `Z.mean(dim=0)`, and for the variance the mean of the squared differences. Careful: in PyTorch, `Z.var(dim=0)` and `Z.std(dim=0)` divide by one row less.",
            "`Z - mean` and dividing by `torch.sqrt(var + 1e-5)` work per column, as standardizing did in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en). `gamma` and `beta` too, because they have one number per column.",
        ],
        why="`mean = Z.mean(dim=0)`, `var = ((Z - mean) ** 2).mean(dim=0)`, and `return gamma * (Z - mean) / torch.sqrt(var + 1e-5) + beta`. In the example, the first column has mean 3 and variance $14/3 \\approx 4.667$, so 1 becomes $(1 - 3)/\\sqrt{4.667} \\approx -0.926$, and with $\\gamma = 1$ and $\\beta = 0$ it stays that.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. An MLP with BatchNorm

        Write a class `BNMLP(nn.Module)`: an MLP with `n_in` inputs, two hidden layers of `n_hidden` neurons and `n_out` logits, with an `nn.BatchNorm1d` between each hidden `nn.Linear` and its ReLU, as in the last line of the recipe. In `__init__`, make the layers in this order: `self.h1 = nn.Linear(n_in, n_hidden)`, `self.bn1`, `self.h2 = nn.Linear(n_hidden, n_hidden)`, `self.bn2` and `self.out = nn.Linear(n_hidden, n_out)`: then the `nn.Linear` layers get the same random weights as in the course's solution. The check makes `BNMLP(4, 6, 3)` after `torch.manual_seed(0)` and compares what it gives for 8 random examples, first in train mode and then in eval mode; then it tries another network after two batches. `.tolist()` turns a tensor into a list of numbers, for the comparison.
        """),
        L.task("Make `BNMLP` an MLP with a BatchNorm1d between each hidden layer and its ReLU."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("bn-mlp", lines=16)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("bn-mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "bn-mlp",
        [
            "In `__init__`, one `nn.BatchNorm1d(n_hidden)` for each hidden layer: each keeps its own running averages and its own $\\gamma$ and $\\beta$.",
            "In `forward`, read each hidden layer from the inside out: first the `nn.Linear`, then its BatchNorm, then `torch.relu`. `self.out` comes last, without ReLU.",
        ],
        why="`__init__`: `super().__init__()`, then `self.h1 = nn.Linear(n_in, n_hidden)`, `self.bn1 = nn.BatchNorm1d(n_hidden)`, `self.h2 = nn.Linear(n_hidden, n_hidden)`, `self.bn2 = nn.BatchNorm1d(n_hidden)` and `self.out = nn.Linear(n_hidden, n_out)`. `forward`: `h = torch.relu(self.bn1(self.h1(x)))`, `h = torch.relu(self.bn2(self.h2(h)))` and `return self.out(h)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The standardizing of [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) now happens inside the network, for every layer and every mini-batch. [Initialization](/course/open/nn-initialization?lang=en) gives a good start; BatchNorm keeps it good during training. In every batch, what BatchNorm gives for a neuron has mean $\beta$ and standard deviation $\gamma$, so as long as $\beta$ is 0 or more, it cannot be 0 or below for every example: the dead layer of [Looking inside a training run](/course/open/nn-activation-stats?lang=en) becomes much less likely. In the next lesson, *Residual connections*, each block adds its input to its output, so that the gradient gets a direct path back through a deep network.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_eval, result_bn, result_mlp):
    L.wrap_up(choice_eval, result_bn, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
