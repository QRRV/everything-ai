# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-deep-net-project. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow residual, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-deep-net-project", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Residual connections](/course/open/nn-residual?lang=en) and [Batch normalization](/course/open/nn-batchnorm?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="A residual block gives `x + f(x)`. Say that the layers of `f` pass back a gradient of almost 0. What reaches `x` on the way back?",
        options={
            "zero": "Almost 0 as well: everything has to go through the layers of `f`",
            "right": "All of the gradient from above: the + passes it on unchanged",
            "twice": "Twice the gradient from above, one copy for each path",
        },
        correct="right",
        why={
            "zero": "The `x` in `x + f(x)` is a second path that skips `f`, and the gradient takes it too. See [Residual connections](/course/open/nn-residual?lang=en).",
            "twice": "The two paths add up: the one past `f` gives the gradient once, the one through `f` almost nothing. See [Residual connections](/course/open/nn-residual?lang=en).",
            "right": "The derivative of `x + f(x)` is 1 plus that of `f`, so the gradient from above always gets through. Today, residual connections are one of the fixes for a network that does not train.",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def quick_check_norm():
    quick_norm = L.choice(
        "quick-norm",
        question="Where does `nn.BatchNorm1d` go in a hidden layer of an MLP?",
        options={
            "before": "Before the `nn.Linear`, on the input of the layer",
            "right": "Between the `nn.Linear` and the ReLU, on its $z$'s",
            "after": "After the ReLU, on what the layer passes on",
        },
        correct="right",
        why={
            "before": "It standardizes what the `nn.Linear` gives, the $z$ of every neuron. See [Batch normalization](/course/open/nn-batchnorm?lang=en).",
            "after": "It standardizes the $z$'s before the ReLU sets the negative ones to 0. See [Batch normalization](/course/open/nn-batchnorm?lang=en).",
            "right": "Linear, then batch normalization, then ReLU: in today's project, it goes into every hidden layer of a deep network.",
        },
    )
    quick_norm
    return (quick_norm,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Depth is what makes networks strong: every layer can build on what the layer before it found. But the deeper the network, the more can go wrong in training, often without an error message. In real projects, a network that does not train well is one of the most common problems, and there is a fixed way to find what is wrong. This project trains a 10-layer MLP on a new dataset, **Fashion-MNIST**: 70,000 small grey pictures of clothes, shoes and bags, 28 by 28 pixels, in 10 kinds, such as trousers, sneakers and ankle boots, made by the web shop Zalando in 2017 as a harder replacement for handwritten digits. The course uses 6,000 of them to train and 1,000 to test.
    """)
    return


@app.cell(hide_code=True)
def predict_plain():
    guess_plain = L.guess(
        "plain-deep",
        question="An MLP with one hidden layer of 64 neurons reads 724 to 807 of the 1,000 test images right after 3 epochs of SGD at a learning rate of 0.1, depending on the seed. Now the same training for an MLP with 10 layers: 64 neurons in every hidden layer, ReLU, and the starting weights of `nn.Linear` itself. How many does it read right?",
        options={
            "more": "About 850: more layers can learn more",
            "less": "About 750: a little worse, as deeper is harder to train",
            "right": "About 100: no better than guessing",
        },
        reveal="**105, for each of 5 seeds.** After training, it says \"trouser\" for every image, and 105 of the 1,000 test images are trousers. Its loss stays at 2.30, the loss of guessing among 10 kinds: $\\ln 10 \\approx 2.30$. Layer after layer, the differences between the images fade: after the eighth hidden layer, they are 700 to 1,000 times smaller than after the first, over 5 seeds, so every image gives the same output.",
    )
    guess_plain
    return (guess_plain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        When a network does not train, first find out whether the problem is in the network and its training step, or somewhere else, such as the data. **Overfit one batch**: train the network on one mini-batch only, the same 64 examples at every step, for a few hundred steps. A network that works learns those 64 by heart, and its loss goes to nearly 0. If even that fails, more data or more epochs will not help: the problem is in the network, its start, or the training step. The test takes seconds, so it comes first.

        The 10 layers fail the test: after 300 steps of SGD at 0.05 on 64 images, the loss is still 2.23. The cause is the spread of the starting weights, from [Initialization](/course/open/nn-initialization?lang=en): those of `nn.Linear` are too small for many ReLU layers, so the signal gets smaller layer after layer, and so does the gradient on the way back. Three fixes from the last lessons each work against that. He initialization gives the weights the right spread from the start. Batch normalization standardizes the $z$'s of every hidden layer over the mini-batch. Residual connections let the signal, and the gradient, pass every layer whole. Then a better optimizer: AdamW with warmup and cosine decay, your Trainer from [A reusable training loop](/course/open/nn-trainer?lang=en). Change one thing at a time, and measure after each: then you know what helped.

        A deep network has many layers, and a plain Python list hides layers from PyTorch: `model.parameters()` only finds the layers that are attributes of the model. `nn.ModuleList` is a list of layers that PyTorch does see. You add a layer with `.append`, and index it and loop over it as a list.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **overfit one batch** | | train on one mini-batch only, to test whether the network and its training step can learn at all | `overfit_one_batch(model, xb, yb, 300, 0.05)` |
        | `nn.ModuleList` | n n module list | a list of layers that PyTorch sees, so that their parameters are trained | `nn.ModuleList()`, then `.append(nn.Linear(64, 64))` |
        """),
        L.watch_out("Overfitting one batch is a test, not a way to train: a network that learned 64 images by heart reads new images badly. And passing the test does not mean the network will do well on all the data: it only rules out the most basic problems. While you look for the cause, change one thing at a time; with two changes at once, you do not know which one helped."),
        mo.md(r"""
        The picture: the same 10 layers, trained for 3 epochs on the 6,000 images, five ways; each bar is one run, with seed 0. Over 5 seeds, the numbers vary: He initialization reads 689 to 731 right, batch normalization 748 to 788, residual connections 718 to 824, and all three with AdamW 838 to 850, more than the MLP with one hidden layer. AdamW alone, without any of the three, gets 624 to 659: Adam scales the tiny gradients up, so this network does learn, but it stays behind each of the three fixes. With the Trainer, one fix is already enough: residual connections alone then read 841 to 851, as much as all three, and the MLP with one hidden layer gets 816 to 824, so the 10 layers win, but by little. And not every mix helps: He initialization with residual connections but without batch normalization makes the signal grow at every block, and with SGD at 0.1 the loss ends as `nan`, for each of 5 seeds.
        """),
        mo.Html(C.fixes_picture({"title": "Test images read right, of 1,000", "plain": "nn.Linear's own starting weights, SGD", "he": "He initialization, SGD",
                                 "norm": "batch normalization, SGD", "residual": "residual connections, SGD", "all": "all three, with the Trainer: AdamW",
                                 "note": "3 epochs on 6,000 images; one run each, seed 0",
                                 "alt": "Five bars of the test images read right, of 1,000: with the starting weights of nn.Linear and SGD 105, with He initialization 690, with batch normalization 787, with residual connections 718, and with all three and the Trainer with AdamW 850."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A deep network that does not train, step by step:

    1. Overfit one batch. If the loss does not go to nearly 0, look at the network and the training step before anything else.
    2. Give the weights the right start: He initialization for ReLU, bias terms at 0.
    3. Add batch normalization to every hidden layer, between the `nn.Linear` and the ReLU.
    4. Add residual connections around the hidden layers.
    5. Train with AdamW, warmup and cosine decay: your Trainer.

    Change one thing at a time, and measure after each.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 for two networks: the 10 layers with the starting weights of `nn.Linear`, and with He initialization, each trained 300 steps on the first 64 training images. `C.DeepMLP` is the course's version of the class you write in exercise 3. The comments with `#` say which step of the recipe each part is. The training step is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_overfit():
    import torch
    import torch.nn.functional as F

    _xb, _yb = C.one_batch()
    # 2. without and with He initialization
    for _he in [False, True]:
        torch.manual_seed(0)
        _model = C.DeepMLP(10, 64, he=_he)
        _opt = torch.optim.SGD(_model.parameters(), lr=0.05)
        # 1. overfit one batch: 300 steps on the same 64 images
        for _step in range(300):
            _loss = F.cross_entropy(_model(_xb), _yb)
            _opt.zero_grad()
            # complete: backward, and the step of _opt
        print(_he, round(_loss.item(), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the loop never takes a step: both losses are the starting loss, 2.3172 for the plain network and 2.481 with He initialization. That is the test failing for another reason, one it is good at finding: a training step that is missing. Completed, with `_loss.backward()` and `_opt.step()`, the plain network ends at 2.2263, close to where it began, and the one with He initialization near 0: it learned the 64 images by heart. The course measured 0.0307; your computer may print another small number there, such as 0.0019, because over 300 steps the smallest rounding difference grows. Over 5 seeds, the plain network always ends between 2.225 and 2.227, and the one with He initialization between 0.002 and 0.031.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overfit_first():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What the test tells you
    """)
    return


@app.cell(hide_code=True)
def choice_overfit_first_cell():
    choice_overfit_first = L.choice(
        "overfit-first",
        question="Your network cannot learn one mini-batch by heart: after 300 steps on the same 64 images, the loss is still 2.23. What does that tell you?",
        options={
            "epochs": "It needs more epochs: 300 steps are too few for 10 layers",
            "data": "The images are too hard: it needs more of them to learn from",
            "right": "The problem is in the network, its start or the training step: not in the data",
        },
        correct="right",
        why={
            "data": "More images cannot help a network that cannot even learn 64 of them by heart: those 64 were all it saw.",
            "epochs": "300 steps are plenty for 64 images: with He initialization the same network gets to 0.03 or lower in those 300 steps.",
            "right": "One mini-batch is the easiest task there is. A network that cannot do it has a problem of its own, such as weights that start too small or a training step that is missing, so you look there before anything else.",
        },
    )
    choice_overfit_first
    return (choice_overfit_first,)


@app.cell(hide_code=True)
def text_exercise_overfit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The overfit test

        Complete `overfit_one_batch(model, xb, yb, steps, lr)`: it trains `model` for `steps` steps of SGD at learning rate `lr`, every step on the same `xb` and `yb`, and gives back the loss of the last step. Replace the `...` with the four lines of a training step, as in the completed worked example. The check trains small models on random data and compares the last loss and the weights. When it is right, the check saves it in your project, as `work/projects/nn/overfit.py`.
        """),
        L.task("Make `overfit_one_batch` train a model on one mini-batch and give the last loss."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overfit():
    box_overfit = L.code_box("overfit", lines=12)
    box_overfit
    return (box_overfit,)


@app.cell(hide_code=True)
def check_overfit(box_overfit):
    result_overfit = L.check("overfit", box_overfit.value)
    result_overfit
    return (result_overfit,)


@app.cell(hide_code=True)
def hints_overfit():
    L.hints(
        "overfit",
        [
            "Four lines, each indented under the `for`: the loss of the model on `xb` and `yb`, then the gradients to 0, then backward, then the step.",
            "The loss is `F.cross_entropy` of the model's output and `yb`. `return loss.item()` after the loop gives the loss of the last round.",
        ],
        why="The loss of every round is computed before its step, so the loss given back is that of the last step, from before its update. `opt.zero_grad()` before `backward()` keeps the gradients of one round from adding up with those before it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_deep():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. One class for every version

        Complete the class `DeepMLP`: `DeepMLP(depth, width, he, norm, residual)` has `depth` layers in all, an input layer, `depth - 2` hidden layers of `width` neurons, and an output layer of 10 logits. `__init__` makes the layers, the hidden ones in an `nn.ModuleList`, with a batch normalization for each when `norm` is `True`. Replace the two `...`:

        - In `__init__`, when `he` is `True`: He initialization for every `nn.Linear`, also `self.inp` and `self.out`, with the bias terms at 0, as in [Initialization](/course/open/nn-initialization?lang=en).
        - In `forward`, the steps of hidden layer `k`: the layer, its batch normalization when there is one, ReLU; then with `residual`, add the result to `h`, and without it, the result replaces `h`.

        The check makes small networks after `torch.manual_seed(0)` and compares what they give for 3 random images, in every version, and the spread of the weights with He initialization. Those are random numbers, so take the layers in the same order as the course: `self.inp`, then the hidden layers, then `self.out`. When it is right, the check saves it as `work/projects/nn/deep_mlp.py`.
        """),
        L.task("Make `DeepMLP` build a deep MLP with He initialization, batch normalization and residual connections as options."),
    ])
    return


@app.cell(hide_code=True)
def exercise_deep():
    box_deep = L.code_box("deep-mlp", lines=34)
    box_deep
    return (box_deep,)


@app.cell(hide_code=True)
def check_deep(box_deep):
    result_deep = L.check("deep-mlp", box_deep.value)
    result_deep
    return (result_deep,)


@app.cell(hide_code=True)
def hints_deep():
    L.hints(
        "deep-mlp",
        [
            "For He initialization: a `for` loop over a list of all `nn.Linear`s, made with `[self.inp] + list(self.hidden) + [self.out]`, with the two lines of [Initialization](/course/open/nn-initialization?lang=en) in it.",
            "In `forward`: first `z = self.hidden[k](h)`; the batch normalization only under `if self.norms is not None:`; then the ReLU; and an `if` and `else` on `self.residual` for what `h` becomes.",
        ],
        why="`list(self.hidden)` turns the `nn.ModuleList` into a list that `+` can join with the other two layers. In `forward`, `h` is what goes on to the next layer: with a residual connection it keeps itself and gets `z` added, so the signal and the gradient can pass every layer whole.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The fading signal of [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) is what stopped the 10 layers. He initialization from [Initialization](/course/open/nn-initialization?lang=en), [Batch normalization](/course/open/nn-batchnorm?lang=en) and [Residual connections](/course/open/nn-residual?lang=en) each fix it, and your Trainer from [A reusable training loop](/course/open/nn-trainer?lang=en), with AdamW, does the rest. Your project now has `overfit_one_batch` and `DeepMLP` next to the Trainer. With *Show extra lessons* on, *A tuning playbook* comes first: there you find a good learning rate with a range test, and then change one hyperparameter at a time, in a small sweep. In the next lesson of the main route, *A neural language model that looks further back*, your language model gets an MLP that reads the last few tokens at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_overfit_first, result_overfit, result_deep):
    L.wrap_up(choice_overfit_first, result_overfit, result_deep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
