# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-dropout. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-dropout", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Regularization and weight decay](/course/open/ml-regularization?lang=en) and [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_worse():
    quick_worse = L.choice(
        "quick-worse",
        question="With regularization, the loss on the training set ends a little higher than without. Why can that still be the better model?",
        options={
            "right": "It learned less of the noise, so it can do better on new examples",
            "faster": "It trained faster, so you can train it for more epochs",
            "smaller": "Its weights became exactly 0, so the model is smaller and quicker to use",
        },
        correct="right",
        why={
            "faster": "Regularization does not make training faster; it changes what the model learns. See [Regularization and weight decay](/course/open/ml-regularization?lang=en).",
            "smaller": "An L2 penalty makes weights smaller, not exactly 0. See [Regularization and weight decay](/course/open/ml-regularization?lang=en).",
            "right": "A model that fits every training example exactly also fits their noise. Regularization gives up a little of the fit on the training set for a better fit on new data. Today you meet a regularizer made for neural networks.",
        },
    )
    quick_worse
    return (quick_worse,)


@app.cell(hide_code=True)
def quick_check_mode():
    quick_mode = L.choice(
        "quick-mode",
        question="A layer you write yourself on top of `nn.Module` must work differently in training than when it predicts. How does its `forward` know which of the two it is doing?",
        options={
            "grad": "It checks whether `torch.no_grad()` is on",
            "eval": "It calls `self.eval()` and looks at what that gives back",
            "right": "It reads `self.training`: `True` in train mode, `False` after `model.eval()`",
        },
        correct="right",
        why={
            "grad": "`torch.no_grad()` only stops the recording of a graph; it says nothing about the mode. See [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en).",
            "eval": "`eval()` switches the mode instead of reading it. See [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en).",
            "right": "`model.eval()` and `model.train()` set `training` in the model and in every layer inside it. Your own dropout layer will read it in exercise 2.",
        },
    )
    quick_mode
    return (quick_mode,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A network with many neurons, trained long on few examples, starts to learn its examples by heart: the loss on the training set keeps falling, the loss on the test set starts to rise. An L2 penalty is one way to hold that back. **Dropout** is a way made for neural networks: during training, it switches off a random part of the neurons in every step. Geoffrey Hinton's group introduced it in 2012, and AlexNet, the network that started the deep-learning boom that year, used it.
    """)
    return


@app.cell(hide_code=True)
def predict_dropout():
    guess_dropout = L.guess(
        "with-dropout",
        question="An MLP with two hidden layers of 256 neurons trains for 100 epochs on only 500 training digits, with SGD at a learning rate of 0.1. It reads 819 of the 1,000 test digits right. Now it gets dropout after each hidden layer: in every training step, each hidden neuron is switched off with a chance of one half. Everything else stays the same. How many test digits do you think it reads right after 100 epochs?",
        options={
            "fewer": "Clearly fewer, about 600: in every step, half of its neurons are missing",
            "right": "A few more, about 840: it overfits less, but 500 digits stay few",
            "many": "Many more, about 950: it no longer overfits at all",
        },
        reveal="**840.** And its loss on the test digits ends lower: 0.60 instead of 0.67. Half the neurons are missing in each step, but each step misses another half, and in the end every neuron has learned. What changes is that no neuron can count on particular others: it has to be useful on its own. That still does not make 500 examples as good as 6,000: the gain is real, but not large.",
    )
    guess_dropout
    return (guess_dropout,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In train mode, dropout sets every number of a layer's output to 0 with chance $p$, each number on its own, with a new random choice at every forward pass. The numbers that stay are divided by $1 - p$. In eval mode, it does nothing: every neuron counts, as it is. A **dropout mask** $m$, a *mask* for short, holds one 0 or 1 for each number. With it, number $i$ of the layer's output, $h_i$, becomes $\tilde{h}_i$:
        """),
        L.formula(r"[[hidden:\tilde{h}_i]] = \frac{m_i}{1 - [[hyper:p]]} \, [[hidden:h_i]]"),
        L.formula(r"m_i = \begin{cases} 1 & \text{with chance } 1 - [[hyper:p]] \\ 0 & \text{with chance } [[hyper:p]] \end{cases}"),
        mo.md(r"""
        Why divide by $1 - p$? With $p = 0.5$, half the numbers drop out, so the next layer gets on average only half of what it gets in eval mode. Doubling the numbers that stay makes up for that: on average, each number is as large in train mode as in eval mode, and the next layer sees the same sizes in both. PyTorch does it this way, during training, so that predicting needs nothing extra.

        Why does it help? Every step trains a different thinned network, made of the neurons that happen to stay, and all of them share the same weights. A neuron cannot count on a particular other neuron being there, so it has to learn something that is useful on its own. In eval mode, the whole network works as a kind of average of all those thinned networks.

        Where and how much: after the activation of a hidden layer, never on the output. The 2014 paper on dropout dropped half of the hidden neurons and a fifth of the inputs; the transformer of 2017 used 0.1. Large language models are often pretrained without dropout: they see most examples only once, and hardly overfit. A relative, DropPath, skips whole blocks of layers at random instead of single numbers; it is used in very deep networks, whose blocks can be skipped. In PyTorch, dropout is the layer `nn.Dropout(p)`; it reads `self.training`, so `model.eval()` switches it off. To write one yourself, `torch.rand_like(x)` gives a tensor in the shape of `x` with random numbers between 0 and 1.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **dropout** | | in train mode, setting each number of a layer's output to 0 with chance $p$ and dividing the rest by $1 - p$ | `nn.Dropout(p)` |
        | $p$ | p | the chance that a number is dropped | `p=0.5` |
        | **dropout mask** | | the zeros and ones that say which numbers stay; a mask for short | `keep = torch.rand_like(x) >= p` |
        | `torch.rand_like` | torch rand like | random numbers between 0 and 1, in the shape of a tensor you give | `torch.rand_like(x)` |
        """),
        L.watch_out("Dropout works only in train mode, so forgetting `model.eval()` before you measure leaves every prediction to chance, and makes the loss and accuracy worse than they are. And dropout slows training down: the training loss falls more slowly, and the network may need more epochs."),
        mo.md(r"""
        The picture: the MLP of the guess, 100 epochs on 500 training digits, without dropout (dashed) and with dropout 0.5 after each hidden layer (solid), and its loss on the 1,000 test digits after each epoch, in eval mode. Without dropout, the test loss is lowest after epoch 30, at 0.56, and then rises to 0.67 at epoch 100: overfitting. With dropout, it falls a little further, to 0.53, and ends at 0.60.
        """),
        mo.Html(C.dropout_picture({"title": "Loss on the test digits", "x": "epoch", "y": "test loss", "without": "without dropout", "with": "with dropout 0.5",
                                   "alt": "Two lines over 100 epochs. Both fall steeply in the first 25 epochs. Without dropout, dashed, the test loss is lowest at about 0.56 around epoch 30 and then rises to about 0.67. With dropout, solid, it stays between about 0.53 and 0.62."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Dropout on the output of a hidden layer:

    1. Choose $p$: often 0.5 for wide hidden layers, 0.1 in transformers.
    2. In train mode: draw a random number between 0 and 1 for every number of the output; keep the numbers where it is at least $p$, and set the others to 0.
    3. Divide the numbers you keep by $1 - p$.
    4. In eval mode: skip steps 2 and 3, and give the output back as it is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Dropout with $p = 0.5$ on the four outputs of a hidden layer, in NumPy: once, then the mean over 10,000 masks, and then in eval mode. `_x * _keep` multiplies by the `True` and `False` of the mask, which count as 1 and 0. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_dropout():
    import numpy as np

    _rng = np.random.default_rng(3)
    _h = np.array([2.0, 1.0, 4.0, 3.0])
    # 1. the chance to drop
    _p = 0.5


    def _drop(_x):
        # 2. a mask: keep where the random number is at least _p
        _keep = _rng.random(4) >= _p
        # 3. complete: also divide by (1 - _p)
        return _x * _keep


    print(_drop(_h))
    _total = np.zeros(4)
    for _i in range(10000):
        _total = _total + _drop(_h)
    print(np.round(_total / 10000, 2))
    # 4. eval mode: no mask, the output as it is
    print(_h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first mask keeps the last two numbers, so as it is, the cell first prints 0, 0, 4 and 3, and then the means 1, 0.5, 2.02 and 1.49: about half of $h$, because half of the numbers drop out. Completed, the first output is 0, 0, 8 and 6, the kept numbers doubled, and the means are 1.99, 1.01, 4.05 and 2.98: about $h$ itself, up to the randomness of 10,000 masks. That is what eval mode gives every time, without a mask: the last line, 2, 1, 4 and 3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Why dropout helps
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-it-helps",
        question="Why does a network with dropout often read new digits better?",
        options={
            "smaller": "It is a smaller network: half of its weights are never trained",
            "faster": "It trains faster, so it reaches a low loss before it can learn the noise of its examples",
            "right": "No neuron can count on particular others, so each learns something useful on its own",
        },
        correct="right",
        why={
            "smaller": "Every weight is trained: each step drops other neurons, and in eval mode all of them count.",
            "faster": "Dropout makes training slower, not faster: each step trains only part of the network, so the loss falls more slowly.",
            "right": "Neurons that only work together with particular others can learn patterns that fit just the training examples. Dropout breaks up those partnerships, and the network as a whole works as an average of many thinned networks.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A dropout layer of your own

        Write a class `Dropout(nn.Module)`, your own `nn.Dropout`. `Dropout(p)` stores `p` as `self.p`. `forward(x)` follows the recipe: in train mode, the mask `keep = torch.rand_like(x) >= self.p`, as in the worked example, and then step 3; `x * keep` sets `x` to 0 where `keep` is `False`. In eval mode, give back `x` as it is; `self.training` says which mode it is. The check sends 100,000 ones through your layer with $p = 0.2$, in train mode and in eval mode, and compares the share of zeros, the mean and the largest number; then the same with $p = 0.5$, and a small tensor with $p = 0$.
        """),
        L.task("Make `Dropout` drop and scale in train mode, and change nothing in eval mode."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("own-dropout", lines=14)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("own-dropout", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "own-dropout",
        [
            "In `__init__`, `self.p = p` after `super().__init__()`. In `forward`, first deal with eval mode: `if not self.training:` give back `x`.",
            "After that comes train mode: the mask, and then `x * keep` divided by `___`, so that the numbers that stay make up for the ones that drop.",
        ],
        why="`__init__`: `super().__init__()` and `self.p = p`. `forward`: `if not self.training: return x`; `keep = torch.rand_like(x) >= self.p`; `return x * keep / (1 - self.p)`. With $p = 0.2$, about 20% of the ones become 0, the others 1.25, and the mean stays about 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The network of the picture

        Write `DropoutMLP(hidden, p)`, the network of the picture: the layers `hidden1`, `hidden2` and `out` are there; add a dropout layer `nn.Dropout(p)` in `__init__`, and use it in `forward` after the ReLU of each hidden layer, not on the output. The check makes `DropoutMLP(8, 0.5)` with a fixed seed, compares its output for a random image in eval mode and in train mode with that of the course's network, and checks that two calls give the same output in eval mode and another one in train mode.
        """),
        L.task("Make `DropoutMLP` use dropout after each hidden layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("dropout-mlp", lines=16)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("dropout-mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "dropout-mlp",
        [
            "One dropout layer is enough for both places: it has no weights, so `self.drop = nn.Dropout(p)` can be used twice.",
            "In `forward`, read each hidden layer from the inside out: first the layer, then `torch.relu`, then the dropout layer. `self.out` comes last, without dropout.",
        ],
        why="`__init__`: `self.drop = nn.Dropout(p)`. `forward`: `h = self.drop(torch.relu(self.hidden1(x)))`; `h = self.drop(torch.relu(self.hidden2(h)))`; `return self.out(h)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Like the L2 penalty of [Regularization and weight decay](/course/open/ml-regularization?lang=en), dropout gives up a little of the fit on the training set for a better fit on new data, but it does so by making training noisier instead of by pulling weights toward 0. It is the layer that [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en) switched on and off: now you know what it does in each mode, and why it scales. In the next lesson, *Embedding layers*, a layer turns ids, such as the number of a word, into vectors that are learned.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why, result_layer, result_mlp):
    L.wrap_up(choice_why, result_layer, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
