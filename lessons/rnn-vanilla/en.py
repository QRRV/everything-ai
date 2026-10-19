# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rnn-vanilla. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rnn-vanilla", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en) and [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="A recurrent network reads a text of 50 characters. How many different matrices $\\mathbf{W}_h$ does it use?",
        options={
            "steps": "50: one for every step",
            "right": "One: the same at every step",
            "between": "49: one between every two steps",
        },
        correct="right",
        why={
            "steps": "A recurrent network uses the same weights at every step, so it can read a text of any length. See [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en).",
            "between": "The hidden state goes from step to step, but always through the same $\\mathbf{W}_h$. See [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en).",
            "right": "One $\\mathbf{W}_h$, used 50 times. This lesson works out its gradient.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_factors():
    quick_factors = L.choice(
        "quick-factors",
        question="On its way back through 30 layers, the gradient is multiplied by a factor of about 0.8 in every layer. It starts at 1. About how large is it at the first layer?",
        options={
            "one": "About 0.8, the factor of one layer",
            "tenth": "About a tenth",
            "right": "About a thousandth",
        },
        correct="right",
        why={
            "one": "Every layer multiplies by its factor, so 30 factors of 0.8 give $0.8^{30}$. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "tenth": "The factors multiply: $0.8^{30}$ is far below 0.1, even though each factor is close to 1. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "right": "$0.8^{30} \\approx 0.0012$: the gradient vanishes, even with factors not far below 1. In this lesson, the layers are the steps of a sequence.",
        },
    )
    quick_factors
    return (quick_factors,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en) you set the weights of a recurrent network by hand. To learn them, training needs their gradients. But $\mathbf{W}_h$ is used at every step of the text, and every step depends on the step before. How do you go back through that? The answer, **backpropagation through time**, also shows why a plain recurrent network forgets what it read long ago, and why the networks of the next lesson were invented.
    """)
    return


@app.cell(hide_code=True)
def predict_reach():
    guess_reach = L.guess(
        "first-input",
        question="A recurrent network with a hidden state of one number reads 30 random inputs, most of them between −1 and 1, with its weight $w = 0.9$. How much does a small change in the first input change the last hidden state, compared with the same change in the last input?",
        options={
            "same": "About as much",
            "power": "About 0.05: that is $0.9^{29}$",
            "right": "Less than a thousandth",
        },
        reveal=f"**Less than a thousandth:** about {L.number(C.reach(0.9)[29], 6)}. Every step back multiplies by $w$, but also by the slope of tanh at that step, which is below 1 wherever the hidden state is not 0. With inputs of this size the hidden state is often far from 0, where that slope is small, so many factors are well below 0.9, and 29 of them leave almost nothing. What the network read 30 steps ago hardly reaches the end, and hardly gets a gradient.",
    )
    guess_reach
    return (guess_reach,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        This lesson takes the smallest recurrent network: a hidden state of one number, $\mathbf{W}_x = 1$ and $\mathbf{b} = 0$. Then $\mathbf{W}_h$ is one weight, $w$, and each step is
        """),
        L.formula(r"[[hidden:h_t]] = \tanh([[param:w]] [[hidden:h_{t-1}]] + [[data:x_t]])"),
        mo.md(r"""
        with $h_0 = 0$. Everything below works for matrices as well, with matrix products and the element-wise product $\odot$ in place of the multiplications.

        **Unroll.** Write the network out over a sequence of $T$ steps, one step after the other. You get a deep network of $T$ layers, one per step, in which every layer has the same weight $w$. That is a computational graph like any other, and the backward pass goes through it from the end to the start, as in [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en). Going back through the steps of a sequence is **backpropagation through time**, or **BPTT**.
        """),
        mo.Html(C.unroll_picture({"title": "One weight, three steps", "note": "forward: the same w at every step",
                                  "alt": "Four circles in a row, h0 to h3, with arrows from each to the next, each marked w. Below h1, h2 and h3 is a box with the input x1, x2 or x3, with an arrow up. Above the circles, arrows in the gradient colour go back from h3 to h0, each marked times w times 1 minus that step's h squared."})),
        mo.md(r"""
        **One step back.** The backward pass starts at the last hidden state, with gradient 1, as if $h_T$ were the loss. At step $t$, the gradient that has come back to $h_t$ goes through tanh, whose slope there is $1 - h_t^2$. Then it splits in two:

        - **Step $t$'s part for $w$.** At step $t$, $w$ multiplies $h_{t-1}$, so this step adds the gradient times $h_{t-1}$ to the gradient for $w$.
        - **On to $h_{t-1}$.** $h_{t-1}$ is multiplied by $w$, so the gradient goes on to the step before times $w$.

        **Add up the parts.** $w$ is used at every step, so its gradient is the sum of the parts of all steps, as the chain rule adds up every path from $w$ to the end:
        """),
        L.formula(r"[[grad:\frac{\partial h_T}{\partial w}]] = \sum_{t=1}^{T} [[grad:\frac{\partial h_T}{\partial h_t}]] \, (1 - [[hidden:h_t]]^2) \, [[hidden:h_{t-1}]]"),
        mo.md(r"""
        **Many factors.** Every step back multiplies the gradient by the factor $w (1 - h_t^2)$: a weight times the slope of an activation function, as each layer did in [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en). A sequence of 100 steps is a network of 100 layers. Factors below 1 make the gradient from far back vanish: the parts of early steps add almost nothing, and the network hardly learns how an early input should change the end. With a matrix $\mathbf{W}_h$ that stretches, the gradient can also explode.

        The picture: how much each input still changes the last hidden state, for 30 random inputs, against how many steps before the end it was read. The y-axis has a log scale: equal steps down are equal factors.
        """),
        mo.Html(C.reach_picture({"title": "How far back the gradient reaches", "x": "steps back from the end", "y": "share that reaches the end",
                                 "w": "w = {w}", "lang": L.lang,
                                 "alt": "Two falling lines on a log scale from 1 down to 10 to the minus 12, against 0 to 29 steps back. With w = 0.9 the line drops to about 0.03 after 10 steps and about 0.00001 after 29. With w = 0.5, dashed, it drops much faster, to about 0.0001 after 10 steps and below 10 to the minus 11 after 29."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **backpropagation through time** | BPTT, B P T T | backpropagation for a recurrent network: unroll it over all steps of the sequence and go back through them | |
        | **unroll** | | write a recurrent network out as one layer per step, all with the same weights | |
        | $\frac{\partial h_T}{\partial h_t}$ | partial h T by partial h t | the gradient that has come back from the end to the hidden state of step $t$ | `grad` |
        | $\frac{\partial h_T}{\partial w}$ | partial h T by partial w | the gradient for $w$: the sum of the parts of all steps | `dw` |
        | **truncated BPTT** | | BPTT that goes back only to the start of a piece of the sequence, such as the last 100 steps | |
        """),
        L.watch_out("BPTT has to keep every hidden state of the forward pass: the backward pass needs $h_t$ and $h_{t-1}$ at every step. So memory grows with the length of the sequence, and long texts are cut into pieces of, say, 100 steps, with the backward pass stopping at the start of each piece: **truncated BPTT**. The hidden state still goes on to the next piece, but no gradient comes back from it, so the network cannot learn anything that reaches back further than a piece."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the gradient of the last hidden state for $w$:

    1. Forward: start at $h_0 = 0$, take every step, and keep every hidden state.
    2. Start the backward pass at the end, with gradient 1, and the gradient for $w$ at 0.
    3. For every step $t$, from the last to the first: multiply the gradient by the slope of tanh, $1 - h_t^2$; add that times $h_{t-1}$ to the gradient for $w$; and multiply it by $w$ to go on to the step before.
    4. After step 1, the gradient for $w$ holds the parts of all steps.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    How much does the first input change the last hidden state, after 3 steps with $w = 0.9$? Going back from $h_3$ to $x_1$ multiplies by the factors of steps 3 and 2, and then by the slope of tanh at step 1. The comments with `#` say which step of the recipe each part is. The factor is not finished: run the cell as it is, then complete it where the comment says, and run it again. PyTorch's autograd, from [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en), gives the number to compare with.
    """)
    return


@app.cell
def example_reach():
    import math
    import torch

    _xs, _w = [0.5, -0.3, 0.8], 0.9
    # 1. forward, keeping every hidden state
    _hs = [0.0]
    for _x in _xs:
        _hs.append(math.tanh(_w * _hs[-1] + _x))
    # 2. the backward pass starts at the end, with gradient 1
    _grad = 1.0
    # 3. back through steps 3 and 2, then through tanh at step 1, to x_1
    for _t in range(len(_xs), 1, -1):
        _grad = _grad * 1.0  # complete: the factor of step t, (1 - _hs[_t] ** 2) * _w
    _grad = _grad * (1 - _hs[1] ** 2)
    print("by hand:", round(_grad, 4))
    # PyTorch's autograd, for comparison
    _xt = torch.tensor(_xs, requires_grad=True)
    _h = torch.tensor(0.0)
    for _t in range(len(_xs)):
        _h = torch.tanh(_w * _h + _xt[_t])
    _h.backward()
    print("PyTorch:", round(float(_xt.grad[0]), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the loop multiplies by 1, and the line `by hand` shows 0.7864: only the slope of tanh at step 1. Completed, it multiplies by $0.9 \times (1 - 0.7182^2) \approx 0.436$ for step 3 and $0.9 \times (1 - 0.1154^2) \approx 0.888$ for step 2, and gives 0.3044, the same as PyTorch. `range(len(_xs), 1, -1)` counts down in steps of −1, from 3 to just above 1: steps 3 and 2. `math.tanh` is the tanh of one number, as `np.tanh` is for an array. The gradient for $w$ goes back the same way, and picks up a part at every step on the way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bptt():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Backpropagation through time

        `bptt(xs, w)` runs the network over the inputs `xs`, one step at a time, keeps every hidden state in `hs`, and gives back the gradient of the last hidden state for `w`. The forward pass is there, and so is the loop back over the steps, which counts `t` down from `len(xs)` to 1. Replace the `...` with the three lines of step 3 of the recipe, with `grad` the gradient that has come back to `hs[t]` and `dw` the gradient for `w`. The check calls `bptt([0.6, -0.2, 0.4], 0.8)`, which should give about 0.4601, and tries one step, a negative `w`, a `w` larger than 1, and `w = 0`.
        """),
        L.task("Make `bptt` go back through every step and add up the parts of the gradient for `w`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bptt():
    box_bptt = L.code_box("bptt", lines=14)
    box_bptt
    return (box_bptt,)


@app.cell(hide_code=True)
def check_bptt(box_bptt):
    result_bptt = L.check("bptt", box_bptt.value)
    result_bptt
    return (result_bptt,)


@app.cell(hide_code=True)
def hints_bptt():
    L.hints(
        "bptt",
        [
            "Three lines: first `local`, the gradient after tanh at step `t`; then add `local` times the hidden state that came into step `t` to `dw`; then the new `grad` for the step before.",
            "`local = grad * (1 - hs[t] ** 2)`. At step `t`, `w` multiplies `hs[t - 1]`: that gives step `t`'s part for `dw`, and the factor on to `hs[t - 1]`.",
        ],
        why="`local = grad * (1 - hs[t] ** 2)`, `dw = dw + local * hs[t - 1]` and `grad = local * w`. For `[0.6, -0.2, 0.4]` with `w = 0.8`, the parts of steps 3, 2 and 1 are 0.164, 0.296 and 0: $h_0 = 0$, so the first step adds nothing.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your function and PyTorch

    The cell below gives five inputs to your `bptt`, and to PyTorch's autograd, which unrolls the same network and goes back through it.
    """)
    return


@app.cell(hide_code=True)
def demo_autograd(box_bptt, result_bptt):
    _both = C.autograd_check(box_bptt.value) if result_bptt.state == "correct" else None
    mo.md(
        f"Your `bptt([0.5, -0.3, 0.8, 0.1, -0.7], 0.9)` gives {L.number(_both[0], 4)}, and PyTorch gives {L.number(_both[1], 4)}: you have done what `backward()` does for a recurrent network. In a real network $w$ is a matrix and the gradient starts at a loss, but the steps are the same."
        if _both else "When your `bptt` is right, it is compared with PyTorch's autograd here."
    )
    return


@app.cell(hide_code=True)
def text_exercise_shared():
    mo.md(r"""
    ### Exercise 2. One weight, many steps
    """)
    return


@app.cell(hide_code=True)
def choice_shared_cell():
    choice_shared = L.choice(
        "shared-grad",
        question="A recurrent network reads 30 steps with one $\\mathbf{W}_h$. What is the gradient of the loss for $\\mathbf{W}_h$?",
        options={
            "last": "The part of the last step: that is where the loss is",
            "right": "The sum of the parts of all 30 steps",
            "mean": "The average of the parts of all 30 steps",
        },
        correct="right",
        why={
            "last": "$\\mathbf{W}_h$ is used at every step, and every use changes the loss. The last step's part leaves out the other 29.",
            "mean": "The chain rule adds up the paths from $\\mathbf{W}_h$ to the loss; it does not divide by their number.",
            "right": "One weight, used 30 times: each use adds its part, as `dw = dw + ...` did in your `bptt`.",
        },
    )
    choice_shared
    return (choice_shared,)


@app.cell(hide_code=True)
def text_exercise_long():
    mo.md(r"""
    ### Exercise 3. A long text
    """)
    return


@app.cell(hide_code=True)
def choice_long_cell():
    choice_long = L.choice(
        "long-text",
        question="A plain recurrent network reads reviews of 200 words. Whether a review is positive is decided by its first word, and the label is read from the last hidden state. What happens in training?",
        options={
            "right": "It hardly learns this: the gradient from the first word vanishes over 199 steps",
            "shared": "It learns this well: the weights are shared, so every step gets the same gradient",
            "explode": "The loss becomes `nan`, because 200 steps always make the gradient explode",
        },
        correct="right",
        why={
            "shared": "The weights are the same at every step, but the gradient is not: from far back it has been multiplied by many factors. See the picture in *The idea*.",
            "explode": "Not always: each factor is a weight times tanh's slope, which is at most 1 and often much smaller, so the factors are mostly below 1 and the gradient vanishes. It explodes only when the factors stretch, above 1.",
            "right": "199 factors $w(1 - h_t^2)$ make the gradient from the first word vanish. The next lesson gives the hidden state a path along which the gradient fades much less.",
        },
    )
    choice_long
    return (choice_long,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The recurrent network of [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en), unrolled into a deep network with the same weight in every layer, and the backward pass of [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en) through it. The factors of [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) are now steps of the sequence: a plain recurrent network forgets because its gradient from far back vanishes. In the next lesson, [LSTM and GRU](/course/open/rnn-lstm?lang=en), gates decide what the hidden state keeps, so that the gradient can reach much further back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long, choice_shared, result_bptt):
    L.wrap_up(result_bptt, choice_shared, choice_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
