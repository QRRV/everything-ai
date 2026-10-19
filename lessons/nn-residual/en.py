# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-residual. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow derivative

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-residual", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) and [Derivative rules](/course/open/math-derivative-rules?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_product():
    quick_product = L.choice(
        "quick-product",
        question="On its way back through 20 layers, the gradient is multiplied by one factor per layer. Every factor is about 0.5. About how large is what reaches the first layer, if it was 1 at the top?",
        options={
            "half": "About 0.5: each layer halves the same gradient once",
            "ten": "About 0.1: twenty halvings shrink it tenfold",
            "right": "About 0.000001: twenty halvings in a row, one per layer",
        },
        correct="right",
        why={
            "half": "The factors multiply: after two layers it is 0.25, after three 0.125. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "ten": "Three or four halvings already make it a tenth. Twenty make it $0.5^{20}$. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "right": "$0.5^{20} \\approx 0.00000095$: the gradient vanishes. Today you see a way to keep every factor close to 1.",
        },
    )
    quick_product
    return (quick_product,)


@app.cell(hide_code=True)
def quick_check_sum_rule():
    quick_sum_rule = L.choice(
        "quick-sum-rule",
        question="$y = x + x^2$. What is the derivative of $y$ with respect to $x$?",
        options={
            "square": "$2x$: the $x$ on its own is a constant",
            "right": "$1 + 2x$: the derivatives of the two terms added",
            "product": "$x \\cdot 2x$: the derivatives of the two terms multiplied",
        },
        correct="right",
        why={
            "square": "The $x$ on its own is not a constant: it changes with $x$, with derivative 1. See [Derivative rules](/course/open/math-derivative-rules?lang=en).",
            "product": "For a sum, the derivatives add up. Multiplying is for a chain of functions. See [Derivative rules](/course/open/math-derivative-rules?lang=en).",
            "right": "The derivative of a sum is the sum of the derivatives: 1 for $x$, and $2x$ for $x^2$. That 1 is the core of today's lesson.",
        },
    )
    quick_sum_rule
    return (quick_sum_rule,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A deep network should be able to do at least what a shallower one does: the extra layers could simply pass their input on. Yet in practice, deeper plain networks often trained worse, also on their own training data, and also with batch normalization. In [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) you saw one reason: on its way back, the gradient is multiplied by one factor per layer, and many factors below 1 leave almost nothing for the first layers. Residual connections changed that in 2015: networks of over 100 layers trained well, and every transformer uses them.
    """)
    return


@app.cell(hide_code=True)
def predict_gradient():
    guess_gradient = L.guess(
        "first-block",
        question="An MLP for the digits has 10 blocks of two layers of 64 neurons, with PyTorch's own starting weights. At the start of training, the gradient that reaches the first block is more than a million times smaller than in the last block. Now each block adds its input to its output; nothing else changes. How large do you think the gradient in the first block is now, compared with the last?",
        options={
            "million": "Still more than a million times smaller",
            "thousand": "About a thousand times smaller",
            "right": "About as large as in the last block",
        },
        reveal="**About as large.** In the last block it is 0.00051, in the first 0.00042. Without the addition, it fell from 0.000038 to 0.000000000025. The addition gives the gradient a direct path back through every block.",
    )
    guess_gradient
    return (guess_gradient,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **residual connection** adds a block's input to its output. The block computes some $f(x)$, such as a layer, a ReLU and another layer, and gives back
        """),
        L.formula(r"y = x + f(x)"),
        mo.md(r"""
        $f$ only has to learn a change to $x$, and with $f(x) = 0$ the block passes $x$ on unchanged: an extra block can always do nothing. On the way back, the derivative of a sum is the sum of the derivatives:
        """),
        L.formula(r"\frac{\partial y}{\partial x} = 1 + \frac{\partial f}{\partial x}"),
        mo.md(r"""
        Through $n$ blocks, the gradient is multiplied by $n$ factors $1 + f'$, not by $n$ factors $f'$. Where $f'$ is small, as with a saturated sigmoid, each factor is still about 1: the gradient no longer vanishes. When $x$ is a vector, as in a layer, the gradient for $y$ comes back to $x$ unchanged, plus what comes back through $f$. In code, the block is one line: `x + f(x)`. It only works when $f$ gives back as many numbers as it gets.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **residual connection** | | adding a block's input to its output: $y = x + f(x)$ | `x + f(x)` |
        | $f(x)$ | f of x | what the block computes from its input, here layer, ReLU, layer | `self.lin2(torch.relu(self.lin1(x)))` |
        """),
        L.watch_out("Not the residual of [How wrong is one prediction?](/course/open/ml-residuals?lang=en): there it is the label minus the prediction; here the word says that the block only learns what to add. And a residual connection does not stop the numbers from growing: every block adds to $x$. In the network of the picture, with a learning rate of 0.3, the loss still becomes `nan` after 20 steps. The normalization of the next lesson helps with that."),
        mo.md(r"""
        The picture: the mean size of the gradient of the weights of the first layer in each block, at the start of training, for an MLP with an input layer, 10 blocks of two layers of 64 neurons and 10 logits, with PyTorch's own starting weights, after one backward pass on 256 training digits. Without residual connections, each block gives $\mathrm{ReLU}(f(h))$, and the gradient falls from 0.000038 in the last block to 0.000000000025 in the first. With them, each block gives $h + f(h)$, and the gradient is between about 0.0004 and 0.0007 in every block. Trained for one epoch with a learning rate of 0.1, the network without them stays at 2.30, the loss of guessing; with them, it reads 838 of the 1,000 test digits right.
        """),
        mo.Html(C.grad_picture({"title": "The gradient in each of 10 blocks, at the start", "x": "block", "y": "mean size of the gradient",
                                "plain": "without residual connections", "res": "with residual connections",
                                "alt": "Two lines on a log scale over 10 blocks. Without residual connections, dashed, the gradient rises from about 0.00000000003 in block 1 to about 0.00004 in block 10. With residual connections, it stays between about 0.0004 and 0.0007."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To give a block a residual connection:

    1. Make $f$ give back as many numbers as it gets: the last layer of $f$ has as many outputs as the block has inputs.
    2. Add the input: $y = x + f(x)$, with nothing after the addition.
    3. Stack the blocks: the input of the first one has a direct path to the last one, forward and back.
    4. To follow the gradient back, go from the last block to the first and multiply by $1 + f'$ for each: 1 for the direct path, plus what comes back through $f$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The chain of [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en): 10 sigmoid layers, each with weight 1, and the input 0.5, where about 0.00000035 came back. Now each layer gives $h + \mathrm{sigmoid}(w h)$. The comments with `#` say which step of the recipe each part is. The comparison at the end is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_residual_chain():
    import numpy as np

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    _w = 1.0
    _h = 0.5
    _zs = []
    # 1. to 3. the forward pass through 10 stacked layers: each adds sigmoid(z) to its input, keeping each z
    for _k in range(10):
        _z = _w * _h
        _zs.append(_z)
        _h = _h + _sigmoid(_z)
    print(np.round(_zs, 3))
    # 4. back from the last layer to the first: times 1 plus the weight times the slope of the sigmoid
    _g = 1.0
    for _z in reversed(_zs):
        _g = _g * (1 + _w * _sigmoid(_z) * (1 - _sigmoid(_z)))
        print(_g)
    # compare: print how many times more comes back than the 0.00000035 without residual connections,
    #    _g / 0.00000035, and run again; then set _w to 4.0 and run once more
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $h$ grows by up to 1 in each layer, so the $z$'s grow too: 0.5, 1.122, 1.877 and on to 8.645. Far from 0, the sigmoid is saturated and its slope is tiny, but each factor is 1 plus that slope, so it never drops below 1. Going back, the gradient is 1.0002 after the last layer and grows to 1.79 at the input: about 5 million times more than without residual connections. With `_w = 4.0`, about 1.44 comes back. Your comparison still divides by the 0.00000035 of weight 1; with weight 4 the plain chain let through only about 0.00000000005, so now about 26 billion times more comes back.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_harmless():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. An extra block
    """)
    return


@app.cell(hide_code=True)
def choice_harmless_cell():
    choice_harmless = L.choice(
        "why-harmless",
        question="You add 10 blocks with residual connections to a network that already trains well. Why does that seldom make it worse?",
        options={
            "skipped": "During training, PyTorch skips blocks that do not help",
            "right": "Each block can learn $f(x) \\approx 0$, and then it simply passes its input on",
            "no-weights": "A residual connection has no weights that can be wrong",
        },
        correct="right",
        why={
            "skipped": "PyTorch runs every block, every time. It is the block itself that can learn to change nothing.",
            "no-weights": "The connection itself has no weights, but $f$ does. What matters is that the block only adds $f(x)$ to its input.",
            "right": "With $f(x)$ close to 0, $y = x + f(x)$ is close to $x$: the block hardly changes anything, and the network does what it did without it. Without the residual connection, a block would have to learn with its weights to pass its input on exactly, which is much harder.",
        },
    )
    choice_harmless
    return (choice_harmless,)


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Back through a residual chain

        Write `residual_chain_gradient(ws, x)`: a chain of layers, one for each weight in the list `ws`, where each layer gives `h + sigmoid(w * h)`, with `h` what the layer before gave, starting with `x`. Give back the gradient that arrives at `x`, starting with 1 at the top, by the recipe. `sigmoid` is already in the box. The check calls `residual_chain_gradient([2.0, 1.0, 1.5], 0.5)`, which should give about `1.747`, and tries other weights and more layers.
        """),
        L.task("Make `residual_chain_gradient` give the gradient that arrives at `x` through the residual chain."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("residual-chain", lines=14)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("residual-chain", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "residual-chain",
        [
            "As in the worked example, but with the weight of each layer from `ws`: in the forward pass, keep each `z` in a list.",
            "On the way back, go through the layers last first, `for k in reversed(range(len(ws))):`, with the weight `ws[k]` and the `zs[k]` of each, and the factor 1 plus the weight times the slope.",
        ],
        why="With the weights 2, 1 and 1.5 and the input 0.5, the $z$'s are 1, 1.231 and 3.008. The factors are $1 + 2 \\times 0.1966 = 1.3932$, $1 + 1 \\times 0.1749 = 1.1749$ and $1 + 1.5 \\times 0.04487 = 1.0673$, and their product is about 1.747.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A block with a residual connection

        Write a class `ResidualBlock(nn.Module)`: `ResidualBlock(d, hidden)` has two layers, `self.lin1 = nn.Linear(d, hidden)` and `self.lin2 = nn.Linear(hidden, d)`, made in that order so that they get the same random weights as in the course's solution. Its `forward(x)` gives `x + f(x)`, with `f` first `lin1`, then ReLU, then `lin2`: the block of the picture. The check makes `ResidualBlock(4, 8)` after `torch.manual_seed(0)` and compares what it gives for 3 random examples; then it tries other sizes, a block applied twice, and whether `x` itself stays as it was.
        """),
        L.task("Make `ResidualBlock` give its input plus f of its input."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("residual-block", lines=12)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("residual-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "residual-block",
        [
            "`lin2` goes from `hidden` back to `d`, so that `f(x)` has as many numbers as `x`, as step 1 of the recipe asks.",
            "In `forward`, read `f` from the inside out: first `self.lin1`, then `torch.relu`, then `self.lin2`. Then add `x`, with nothing after it.",
        ],
        why="`__init__`: `super().__init__()`, `self.lin1 = nn.Linear(d, hidden)` and `self.lin2 = nn.Linear(hidden, d)`. `forward`: `return x + self.lin2(torch.relu(self.lin1(x)))`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The product of factors from [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) is still there, but every factor is now $1 + f'$: the sum rule from [Derivative rules](/course/open/math-derivative-rules?lang=en) gives each block a direct path back. Your `ResidualBlock` is the shape of a transformer block: there, $f$ is attention or an MLP, and the input runs past both. In the next lesson, *Layer normalization*, you standardize each example over its own features, and see where that goes in such a block.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_harmless, result_chain, result_block):
    L.wrap_up(choice_harmless, result_chain, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
