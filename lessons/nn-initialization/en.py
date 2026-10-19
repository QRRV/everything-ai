# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-initialization. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-initialization", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Common distributions](/course/open/prob-distributions?lang=en) and [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_randn():
    quick_randn = L.choice(
        "quick-randn",
        question="`random.gauss(0, 0.1)` draws one number. From which distribution?",
        options={
            "right": "The normal distribution with mean 0 and standard deviation 0.1: $\\mathcal{N}(0, 0.1^2)$",
            "variance": "The normal distribution with mean 0 and variance 0.1: $\\mathcal{N}(0, 0.1)$",
            "uniform": "Every number from −0.1 to 0.1 equally likely: a uniform spread",
        },
        correct="right",
        why={
            "variance": "`random.gauss` takes the standard deviation, not the variance; the variance then is $0.1^2 = 0.01$. See [Common distributions](/course/open/prob-distributions?lang=en).",
            "uniform": "`gauss` draws from the normal distribution, the bell curve, not evenly. See [Common distributions](/course/open/prob-distributions?lang=en).",
            "right": "`random.gauss(mu, sigma)` takes the standard deviation, and formulas write the variance: $\\mathcal{N}(0, 0.1^2)$. Starting weights are drawn the same way.",
        },
    )
    quick_randn
    return (quick_randn,)


@app.cell(hide_code=True)
def quick_check_saturated():
    quick_saturated = L.choice(
        "quick-saturated",
        question="A neuron with a sigmoid gets $z = 8$. Why does it learn almost nothing?",
        options={
            "zero": "Its output is 0: nothing comes through",
            "right": "It is saturated: its slope there is almost 0",
            "big": "Its gradient is too large: it explodes",
        },
        correct="right",
        why={
            "zero": "At $z = 8$ the sigmoid gives almost 1, not 0. What matters on the way back is its slope. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "big": "Far from 0 the sigmoid is flat: its slope is tiny, not large. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "right": "$\\text{sigmoid}(8) \\approx 0.9997$, and the slope $\\text{sigmoid}(8)(1 - \\text{sigmoid}(8)) \\approx 0.0003$: whatever comes back is multiplied by that.",
        },
    )
    quick_saturated
    return (quick_saturated,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en), the spread of the starting weights decided whether the gradient of a deep network vanished, exploded or stayed the same size. Every network starts somewhere: `nn.Linear` draws small random numbers for you. Which spread is right depends on how many inputs a neuron has and on its activation function. With the wrong one, a deep network hardly learns, however long you train it; with the right one, it starts in a place from which gradient descent can go somewhere.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "all-equal",
        question="Every weight of an MLP with a hidden layer of 64 neurons starts at the same number, 0.1, and every bias term at 0. You train it for a long time. How many of the 64 hidden neurons do you think end up doing something different from the others?",
        options={
            "all": "All 64: training pulls them apart",
            "half": "About half: ReLU switches some of them off",
            "right": "None: they stay copies of each other",
        },
        reveal="**None.** Every hidden neuron gets the same inputs and has the same weights, so it gives the same output. On the way back, every one gets the same gradient, because the weights after them are equal too, and so the same step. After the step they are equal again, and so after every step: 64 copies of one neuron. Random starting weights break that symmetry.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Starting weights are random, so that the neurons of a layer start different and stay different. But how spread out? A neuron adds up $n_\text{in}$ inputs, each times a weight: $z = \sum_j w_j x_j$. When the weights and the inputs are independent, so that knowing one tells you nothing about the other, and have mean 0, the variances multiply and add up; this lesson takes that rule from probability as given:
        """),
        L.formula(r"\mathrm{Var}(z) = n_\text{in} \cdot \mathrm{Var}([[param:w]]) \cdot \mathrm{Var}(x)"),
        mo.md(r"""
        To keep the signal the same size, $\mathrm{Var}(w)$ must be $1 / n_\text{in}$. ReLU then sets about half of the $z$'s to 0, which halves what comes through; so for ReLU, the variance is doubled. That is **He initialization**, after Kaiming He. **Xavier initialization**, after Xavier Glorot, counts the outputs $n_\text{out}$ as well, so that the gradient on the way back keeps its size too. It is meant for activation functions that let the whole signal through around 0, such as tanh: an S-shaped curve from −1 to 1, like a sigmoid with 0 in the middle, with slope 1 at 0. Recurrent neural networks, later in the course, use it.
        """),
        L.formula(r"\text{He: } \sigma = \sqrt{\frac{2}{n_\text{in}}}"),
        L.formula(r"\text{Xavier: } \sigma = \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}"),
        mo.md(r"""
        The weights are drawn from $\mathcal{N}(0, \sigma^2)$ with that $\sigma$, and the bias terms start at 0: the random weights already take care of **symmetry breaking**. In PyTorch, `nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")` does He, and `nn.init.xavier_normal_(layer.weight)` Xavier; `nn.init.zeros_(layer.bias)` sets the bias terms to 0. The `_` at the end of their names means: they change the tensor itself. `nn.Linear` draws its own starting weights evenly from a smaller range, with a standard deviation of about $1/\sqrt{3 n_\text{in}}$: enough for a few layers, too small for many ReLU layers.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **initialization** | | choosing the starting values of the weights and bias terms, before training | `nn.init` |
        | **symmetry breaking** | | random starting weights, so that the neurons of a layer start different and can learn different things | |
        | **He initialization** | | starting weights from $\mathcal{N}(0, \sigma^2)$ with $\sigma = \sqrt{2 / n_\text{in}}$: for ReLU | `nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")` |
        | **Xavier initialization** | | starting weights with $\sigma = \sqrt{2 / (n_\text{in} + n_\text{out})}$: for tanh | `nn.init.xavier_normal_(layer.weight)` |
        | $n_\text{in}$, $n_\text{out}$ | n in, n out | the number of inputs and of outputs of a layer | `layer.in_features`, `layer.out_features` |
        | `nn.init.zeros_` | n n init zeros | sets a tensor to 0 in place | `nn.init.zeros_(layer.bias)` |
        | `torch.randn` | torch rand n | a tensor of random numbers from $\mathcal{N}(0, 1)$, in the shape you give | `torch.randn(512, 256)` |
        | **activation** | | what a neuron or layer gives out, after its activation function | `h` |
        """),
        L.watch_out("He and Kaiming are the same method, named after Kaiming He; Xavier and Glorot are the same too, after Xavier Glorot. And the spread is only for the start: training moves the weights wherever the gradients send them. Other ways exist too, such as orthogonal initialization; He and Xavier are the ones you meet most. Layers that normalize the signal, later in the course, make the start matter less."),
        mo.md(r"""
        The picture: the standard deviation of the activations after each of 20 ReLU layers of 256 neurons, for 512 random inputs, with He starting weights and with Xavier starting weights. With He, the signal keeps its size; with Xavier, made for tanh, it loses about 30% in every ReLU layer.
        """),
        mo.Html(C.spread_picture({"title": "The spread of the activations through 20 ReLU layers", "x": "layer", "y": "standard deviation",
                                  "he": "He", "xavier": "Xavier",
                                  "alt": "Two lines on a log scale over 20 layers. With He, the standard deviation stays between about 0.55 and 0.87. With Xavier, dashed, it falls from about 0.6 to about 0.0006."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To give a network starting weights:

    1. Choose the spread for the activation function: He for ReLU, Xavier for tanh.
    2. Count the inputs of each layer, and for Xavier its outputs too, and work out $\sigma$.
    3. Draw the weights from $\mathcal{N}(0, \sigma^2)$, and set the bias terms to 0.
    4. Check: send random inputs through, and look at the spread of the activations after each layer. It should stay about the same.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    20 ReLU layers of 256 inputs and 256 outputs, with He starting weights. The comments with `#` say which step of the recipe each part is. The last comment is for you: run the cell as it is, then do what it says, and run it again.
    """)
    return


@app.cell
def example_init():
    import torch
    from torch import nn

    torch.manual_seed(0)
    _layers = []
    for _i in range(20):
        _layers.append(nn.Linear(256, 256))
    # 1. and 2. ReLU, so He: sigma is the square root of 2 / 256
    print((2 / 256) ** 0.5)
    # 3. the weights from N(0, sigma²), the bias terms 0
    for _layer in _layers:
        nn.init.kaiming_normal_(_layer.weight, nonlinearity="relu")
        nn.init.zeros_(_layer.bias)
    print(_layers[0].weight.std().item())
    # 4. 512 random inputs through all layers, with the spread after each; printed after layers 1, 5, 10 and 20
    _h = torch.randn(512, 256)
    _spreads = []
    with torch.no_grad():
        for _layer in _layers:
            _h = torch.relu(_layer(_h))
            _spreads.append(_h.std().item())
    print(_spreads[0], _spreads[4], _spreads[9], _spreads[19])
    # then: replace kaiming_normal_ in step 3 by nn.init.xavier_normal_(_layer.weight), and run again
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\sqrt{2/256} \approx 0.0884$, and the drawn weights of the first layer have about that spread: 0.0886. The activations stay about the same size: 0.83 after layer 1, 0.87 after layer 5, 0.69 after layer 10 and 0.62 after layer 20. With `xavier_normal_`, the spread is $\sqrt{2/512} \approx 0.0625$, and the drawn weights have 0.0627: too small for ReLU. The activations fall from 0.59 after layer 1 to 0.15 after layer 5, 0.021 after layer 10 and 0.0006 after layer 20.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_he():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. He for 800 inputs
    """)
    return


@app.cell(hide_code=True)
def choice_he_cell():
    choice_he = L.choice(
        "he-512",
        question="A ReLU layer has 800 inputs and 200 outputs. Which standard deviation does He initialization give its weights?",
        options={
            "one": "About 0.035: the square root of 1/800",
            "right": "About 0.05: the square root of 2/800",
            "xavier": "About 0.045: the square root of 2/(800 + 200)",
        },
        correct="right",
        why={
            "one": "That keeps the signal for an activation function that lets it all through. ReLU halves it, so He doubles the variance.",
            "xavier": "That is Xavier, which counts the outputs too; it is meant for tanh. He counts only the inputs.",
            "right": "$\\sqrt{2/800} = \\sqrt{1/400} = 1/20 = 0.05$: He counts only the inputs, and doubles the variance for ReLU.",
        },
    )
    choice_he
    return (choice_he,)


@app.cell(hide_code=True)
def text_exercise_std():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The spread as a function

        Write `init_std(n_in, n_out, activation)`: the standard deviation of the starting weights of a layer with `n_in` inputs and `n_out` outputs, by He when `activation` is `"relu"` and by Xavier when it is `"tanh"`. The check calls `init_std(200, 100, "relu")`, which should give `0.1`, and tries other layers and `"tanh"`.
        """),
        L.task("Make `init_std` give the spread of the starting weights by He or by Xavier."),
    ])
    return


@app.cell(hide_code=True)
def exercise_std():
    box_std = L.code_box("init-std", lines=6)
    box_std
    return (box_std,)


@app.cell(hide_code=True)
def check_std(box_std):
    result_std = L.check("init-std", box_std.value)
    result_std
    return (result_std,)


@app.cell(hide_code=True)
def hints_std():
    L.hints(
        "init-std",
        [
            "Two cases: `if activation == \"relu\":` gives back one spread, and otherwise the other.",
            "Both spreads are a square root, `** 0.5`, of 2 divided by a count: for He the inputs, for Xavier the inputs plus the outputs.",
        ],
        why="He for ReLU: $\\sqrt{2/200} = \\sqrt{0.01} = 0.1$. Xavier for tanh would give $\\sqrt{2/300} \\approx 0.0816$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_init():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. He starting weights for a whole network

        Write `he_init(layers)`: it gets a list of `nn.Linear` layers and gives every one He starting weights and bias terms of 0, by step 3 of the recipe; it changes the layers themselves and gives nothing back. The check makes two layers, `nn.Linear(400, 100)` and `nn.Linear(100, 50)`, after `torch.manual_seed(0)`, calls `he_init` on them, and looks at the spread of each layer's weights, about 0.071 and 0.141, and at its bias terms; then at three other layers. Random weights differ a little with another way of drawing them, so the check allows a few percent.
        """),
        L.task("Make `he_init` give every layer He starting weights and bias terms of 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_init():
    box_init = L.code_box("he-init", lines=8)
    box_init
    return (box_init,)


@app.cell(hide_code=True)
def check_init(box_init):
    result_init = L.check("he-init", box_init.value)
    result_init
    return (result_init,)


@app.cell(hide_code=True)
def hints_init():
    L.hints(
        "he-init",
        [
            "A loop over the layers, as in step 3 of the worked example.",
            "`nn.init.kaiming_normal_` takes the number of inputs from the shape of the weights itself, so the same line works for every layer.",
        ],
        why="`for layer in layers:` with `nn.init.kaiming_normal_(layer.weight, nonlinearity=\"relu\")` and `nn.init.zeros_(layer.bias)`. The first layer gets $\\sigma = \\sqrt{2/400} \\approx 0.071$, the second $\\sqrt{2/100} \\approx 0.141$.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The variance of [Expected value and variance](/course/open/prob-expectation?lang=en) and the normal distribution of [Common distributions](/course/open/prob-distributions?lang=en) give the spread that keeps the signal the same size; He doubles it for ReLU, which halves the signal. That keeps the gradients of [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) the same size too, at least at the start. In the next lesson, *Looking inside a training run*, you look inside a training run, at the activations and gradients of every layer, and see where it goes wrong.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_he, result_std, result_init):
    L.wrap_up(choice_he, result_std, result_init)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
