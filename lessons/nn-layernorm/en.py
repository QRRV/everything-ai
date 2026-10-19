# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-layernorm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow residual

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-layernorm", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Batch normalization](/course/open/nn-batchnorm?lang=en) and [Residual connections](/course/open/nn-residual?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_over_what():
    quick_over_what = L.choice(
        "quick-over-what",
        question="In train mode, batch normalization standardizes the $z$'s of each neuron. Over what does it take the mean and the variance?",
        options={
            "all-data": "Over all the training data at once, before training starts",
            "right": "Over the examples in the mini-batch",
            "running": "Over its running averages, which it updates after every batch",
        },
        correct="right",
        why={
            "all-data": "It never sees all the data at once: in train mode, it uses the mini-batch it gets. See [Batch normalization](/course/open/nn-batchnorm?lang=en).",
            "running": "The running averages are for eval mode. In train mode, it uses the mean and variance of the mini-batch itself. See [Batch normalization](/course/open/nn-batchnorm?lang=en).",
            "right": "Per neuron, over the examples in the mini-batch: so each answer depends on the other examples in it.",
        },
    )
    quick_over_what
    return (quick_over_what,)


@app.cell(hide_code=True)
def quick_check_direct_path():
    quick_direct_path = L.choice(
        "quick-direct-path",
        question="A block gives $y = x + f(x)$. What comes back to $x$ on the way back?",
        options={
            "only-f": "Only what comes back through $f$, as in any layer",
            "nothing": "Nothing: the addition stops the gradient",
            "right": "The gradient for $y$ itself, plus what comes back through $f$",
        },
        correct="right",
        why={
            "only-f": "The addition gives $x$ a second path, with derivative 1. See [Residual connections](/course/open/nn-residual?lang=en).",
            "nothing": "An addition passes the gradient on to both of its terms. See [Residual connections](/course/open/nn-residual?lang=en).",
            "right": "$\\partial y / \\partial x = 1 + \\partial f / \\partial x$: the direct path brings the gradient for $y$ back unchanged. Today you see what may and may not sit on that path.",
        },
    )
    quick_direct_path
    return (quick_direct_path,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Batch normalization needs a batch: each answer depends on the other examples in it, small batches give rough estimates, and eval mode needs running averages. A language model that writes text works on one sequence at a time, of any length. Layer normalization standardizes each example on its own, over its own features, so none of that matters. Transformers use it in every block. That raises a second question: in a block with a residual connection, where does the normalization go?
    """)
    return


@app.cell(hide_code=True)
def predict_where():
    guess_where = L.guess(
        "post-or-pre",
        question="An MLP for the digits has 30 blocks with residual connections, and trains for one epoch with a learning rate of 0.3. Without normalization, its loss is `nan` after 6 steps. Now each block gets a layer normalization: either after the addition, as in the first transformer of 2017, or at the start of $f$, with one more after the last block. Which do you think trains?",
        options={
            "both": "Both, about equally well: a LayerNorm is a LayerNorm",
            "after": "Only the one with the LayerNorm after the addition",
            "right": "Only the one with the LayerNorm at the start of $f$",
        },
        reveal="**Only the one with the LayerNorm at the start of $f$.** After one epoch it reads 784 of the 1,000 test digits right. With the LayerNorm after the addition, the loss stays at 2.30, the loss of guessing, and it reads 126 right. Every LayerNorm on the direct path changes what comes back through it; in 30 blocks, that is 30 changes in a row.",
    )
    guess_where
    return (guess_where,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Layer normalization**, LayerNorm for short, standardizes each example over its own features. For one example with $d$ features $x_1, \dots, x_d$:
        """),
        L.formula(r"\mu = \frac{1}{d} \sum_{j=1}^{d} x_j \qquad \sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2"),
        L.formula(r"\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + [[hyper:\varepsilon]]}} \qquad y_j = [[param:\gamma_j]] \, \hat{x}_j + [[param:\beta_j]]"),
        mo.md(r"""
        The formulas are those of batch normalization, turned around: the mean and variance are taken over the $d$ features of one example, not over the examples of a batch. $\gamma$ and $\beta$ still have one number per feature, learned, the same for every example. There are no running averages: LayerNorm does the same in train mode and in eval mode, and it works for a batch of one. In PyTorch, `nn.LayerNorm(64)` does this for 64 features.

        In a block with a residual connection, it can go in two places. **Post-norm**, as in the first transformer of 2017, normalizes after the addition: $y = \mathrm{LN}(x + f(x))$. **Pre-norm**, as in GPT-2 and most transformers since, normalizes on the way into $f$: $y = x + f(\mathrm{LN}(x))$. With pre-norm, nothing sits on the direct path, so the gradient comes back through every block unchanged, as in [Residual connections](/course/open/nn-residual?lang=en). Because the path itself is never normalized, its numbers can grow from block to block; a pre-norm network therefore puts one more LayerNorm after the last block, before the output layer. Without it, the pre-norm network of the guess ends in `nan` too.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **layer normalization** | | for each example: standardize its features, then times $\gamma$ plus $\beta$ | `nn.LayerNorm(64)` |
        | $\mu$, $\sigma^2$ | mu, sigma squared | the mean and the variance of one example's features, dividing by $d$ | `X.mean(dim=1, keepdim=True)` |
        | **post-norm** | | the LayerNorm after the addition: $\mathrm{LN}(x + f(x))$ | `self.norm(x + f(x))` |
        | **pre-norm** | | the LayerNorm on the way into $f$: $x + f(\mathrm{LN}(x))$ | `x + f(self.norm(x))` |
        | $\mathrm{LN}(x)$ | L N of x | layer normalization as a function: $x$ standardized over its features, times $\gamma$ plus $\beta$ | `self.norm(x)` |
        | `keepdim=True` | keep dim true | keeps the mean of each row as a column, so that it can be taken from every number in the row | `X.mean(dim=1, keepdim=True)` |
        """),
        L.watch_out("`keepdim` in PyTorch, `keepdims` in NumPy: the same thing, with and without an s. And post-norm is not wrong: with a smaller learning rate, or one that grows slowly at the start, it trains well too. With 0.1 instead of 0.3, the 30-block network reads about 84% of the test digits with either. Pre-norm is the choice that needs the least care."),
        mo.md(r"""
        The picture: the training loss of the 30-block network of the guess during its one epoch, the mean of the last 10 mini-batches, with post-norm and with pre-norm. In the first 20 steps, both rise above 3 and fall back. Then, with post-norm, the loss stays at 2.30. With pre-norm, it keeps falling, to about 0.7.
        """),
        mo.Html(C.loss_picture({"title": "30 blocks, learning rate 0.3, one epoch", "x": "mini-batch", "y": "training loss",
                                "post": "post-norm", "pre": "pre-norm",
                                "alt": "Two loss curves over 94 mini-batches. Both rise from about 2.6 to above 3 in the first 20 steps and fall back. With post-norm, dashed, the loss then stays at about 2.3. With pre-norm, it falls to about 0.7."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Layer normalization of a batch, one row per example and one column per feature:

    1. Per row: the mean and the variance over its features, dividing by the number of features, and kept as a column with `keepdim=True`.
    2. Standardize: minus the mean, divided by the square root of the variance plus $\varepsilon$.
    3. Rescale: times $\gamma$, plus $\beta$, one of each per feature.
    4. In a block with a residual connection, use pre-norm: $x + f(\mathrm{LN}(x))$, and one more LayerNorm before the output layer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The batch of [Batch normalization](/course/open/nn-batchnorm?lang=en), turned around: now 2 examples with 4 features each, the first between 1 and 6, the second between 40 and 70. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_layernorm():
    import torch
    from torch import nn

    _X = torch.tensor([[1.0, 2.0, 3.0, 6.0], [50.0, 70.0, 40.0, 60.0]])
    # 1. per row: the mean and the variance over its features, kept as a column
    _mean = _X.mean(dim=1, keepdim=True)
    _var = ((_X - _mean) ** 2).mean(dim=1, keepdim=True)
    print(_mean, _var)
    # 2. and 3. standardize; gamma and beta start at 1 and 0, so rescaling changes nothing yet
    print((_X - _mean) / torch.sqrt(_var + 1e-5))
    _ln = nn.LayerNorm(4)
    with torch.no_grad():
        print(_ln(_X))
        print(_ln(_X[:1]))
        # 4. a pre-norm block, with f one layer: _X plus f of the normalized _X
        torch.manual_seed(0)
        _f = nn.Linear(4, 4)
        # then: print _X plus _f of _ln(_X), and run again
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The means are 3 and 55, the variances 3.5 and 125: the numbers of Batch normalization, because the batch is the same, turned around. Each row now has mean 0 and standard deviation 1: −1.07, −0.53, 0 and 1.60 for the first example, −0.45, 1.34, −1.34 and 0.45 for the second. `nn.LayerNorm(4)` gives the same. Given only the first example, it gives exactly the same first row: the second example has no influence. With BatchNorm, a batch of one would have been standardized to all zeros. In step 4, $f$ adds at most about 1 to each number: the first row becomes 0.47, 3.07, 2.48 and 6.50, and the second stays between 40 and 71. The direct path carries `_X` itself; only $f$ gets the normalized rows.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A batch of one
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "batch-of-one",
        question="A model trains on one long text per step: a batch of one. In train mode, which normalization still standardizes its features in a useful way?",
        options={
            "batch": "BatchNorm: its running averages fill in for the missing batch",
            "right": "LayerNorm: it standardizes each example over its own features",
            "both": "Both: a batch of one is just a small batch",
        },
        correct="right",
        why={
            "batch": "In train mode, BatchNorm uses the batch itself, not its running averages. For one example, each $z$ minus its own mean is 0: everything becomes $\\beta$.",
            "both": "For BatchNorm, the mean of one number is that number, so every $z$ becomes 0 before rescaling; PyTorch even refuses a batch of one in train mode. LayerNorm never looks at other examples.",
            "right": "LayerNorm takes the mean and variance over the features of each example, so the size of the batch does not matter, in train mode or in eval mode.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_ln():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Layer normalization by hand

        Write `layer_norm(X, gamma, beta)`: layer normalization, by steps 1 to 3 of the recipe, with $\varepsilon$ = 0.00001. `X` has one row per example and one column per feature; `gamma` and `beta` have one number per feature. The check calls `layer_norm(torch.tensor([[1.0, 2.0, 6.0], [10.0, 20.0, 30.0]]), torch.tensor([1.0, 2.0, 0.5]), torch.tensor([0.0, 1.0, -1.0]))`, which should give `[[-0.9258, 0.0742, -0.3056], [-1.2247, 1.0, -0.3876]]`, and tries other batches, also one of a single example.
        """),
        L.task("Make `layer_norm` standardize each row over its own features, then times gamma, plus beta."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ln():
    box_ln = L.code_box("layer-norm", lines=6)
    box_ln
    return (box_ln,)


@app.cell(hide_code=True)
def check_ln(box_ln):
    result_ln = L.check("layer-norm", box_ln.value)
    result_ln
    return (result_ln,)


@app.cell(hide_code=True)
def hints_ln():
    L.hints(
        "layer-norm",
        [
            "As `batch_norm` in [Batch normalization](/course/open/nn-batchnorm?lang=en), but over each row instead of each column: `dim=1`.",
            "With `dim=1`, the mean is one number per row; `keepdim=True` keeps it as a column, so that `X - mean` takes each row's own mean from that row.",
        ],
        why="`mean = X.mean(dim=1, keepdim=True)`, `var = ((X - mean) ** 2).mean(dim=1, keepdim=True)`, and `return gamma * (X - mean) / torch.sqrt(var + 1e-5) + beta`. In the example, the first row has mean 3 and variance $14/3 \\approx 4.667$, so its 1 becomes $(1 - 3)/\\sqrt{4.667} \\approx -0.926$, times 1 plus 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A pre-norm block

        Write a class `PreNormBlock(nn.Module)`: your `ResidualBlock` of [Residual connections](/course/open/nn-residual?lang=en) with pre-norm. `PreNormBlock(d, hidden)` has `self.norm = nn.LayerNorm(d)`, `self.lin1 = nn.Linear(d, hidden)` and `self.lin2 = nn.Linear(hidden, d)`, with `lin1` made before `lin2` so that they get the same random weights as in the course's solution. Its `forward(x)` gives `x + f(norm(x))`, with `f` first `lin1`, then ReLU, then `lin2`. The check makes `PreNormBlock(4, 8)` after `torch.manual_seed(0)` and compares what it gives for 3 random examples; then it tries other sizes, a block applied twice, and a single example.
        """),
        L.task("Make `PreNormBlock` give its input plus f of its normalized input."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("pre-norm-block", lines=13)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("pre-norm-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "pre-norm-block",
        [
            "`__init__` is that of your `ResidualBlock`, with `self.norm = nn.LayerNorm(d)` added: `d` features, the size of `x`.",
            "In `forward`, only `f` gets the normalized input: first `self.norm`, then `lin1`, ReLU and `lin2`. The `x` that is added stays as it is.",
        ],
        why="`__init__`: `super().__init__()`, `self.norm = nn.LayerNorm(d)`, `self.lin1 = nn.Linear(d, hidden)` and `self.lin2 = nn.Linear(hidden, d)`. `forward`: `return x + self.lin2(torch.relu(self.lin1(self.norm(x))))`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The formulas of [Batch normalization](/course/open/nn-batchnorm?lang=en), taken over the features of each example instead of over the batch. The direct path of [Residual connections](/course/open/nn-residual?lang=en) stays clean with pre-norm, so that the gradient keeps coming back unchanged. Your `PreNormBlock` is half of a transformer block: there, one such block has attention as its $f$, and the next one an MLP. In the next lesson, *Momentum*, gradient descent itself gets a memory of its earlier steps.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, result_ln, result_block):
    L.wrap_up(choice_one, result_ln, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
