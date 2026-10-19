# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rnn-lstm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rnn-lstm", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A recurrent network and backprop through time](/course/open/rnn-vanilla?lang=en) and [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_factor():
    quick_factor = L.choice(
        "quick-factor",
        question="In a recurrent network with a hidden state of one number, $h_t = \\tanh(w h_{t-1} + x_t)$, what does one step back multiply the gradient by?",
        options={
            "right": "$w (1 - h_t^2)$",
            "part": "$(1 - h_t^2) \\, h_{t-1}$",
            "weight": "$w$",
        },
        correct="right",
        why={
            "weight": "The gradient also goes through tanh, whose slope is below 1 almost everywhere. See [A recurrent network and backprop through time](/course/open/rnn-vanilla?lang=en).",
            "part": "That factor gives step $t$'s part for $w$: there, $w$ multiplies $h_{t-1}$. On to $h_{t-1}$, the gradient passes $w$ instead. See [A recurrent network and backprop through time](/course/open/rnn-vanilla?lang=en).",
            "right": "A weight times the slope of tanh, once per step: over many steps that product vanishes. This lesson changes the path the gradient takes.",
        },
    )
    quick_factor
    return (quick_factor,)


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Which numbers can the sigmoid $\\sigma(z)$ give?",
        options={
            "tanh": "Between −1 and 1",
            "right": "Between 0 and 1",
            "any": "Any number, like $z$ itself",
        },
        correct="right",
        why={
            "tanh": "That is tanh. The sigmoid turns a score into a probability. See [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "any": "The sigmoid squeezes any $z$ into a number between 0 and 1. See [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "right": "Between 0 and 1: multiply a number by it, and you keep a share of that number. That is what a gate does.",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A plain recurrent network forgets: the gradient from far back vanishes, so it hardly learns what an early word should do to the end. The **LSTM**, from 1997, was made for exactly this problem. Before transformers took over, from 2017 on, LSTMs translated texts, recognised speech and predicted the next word on phone keyboards; they still run where a model must read a long stream one step at a time. The **GRU** is a simpler version that often works about as well.
    """)
    return


@app.cell(hide_code=True)
def predict_keep():
    guess_keep = L.guess(
        "keep-100",
        question="An LSTM keeps a second vector next to its hidden state. Each step, one number of it is multiplied by 0.99, and nothing is added. How much of that number is left after 100 steps?",
        options={
            "all": "All of it",
            "right": "About a third",
            "nothing": "Almost nothing, as in a plain recurrent network",
        },
        reveal="**About a third:** $0.99^{100} \\approx 0.37$. That is a lot: in a plain recurrent network, less than a thousandth of the gradient reached 30 steps back. The gradient back along this vector is multiplied by the same 0.99 per step, so it reaches far back too.",
    )
    guess_keep
    return (guess_keep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A second memory.** An LSTM carries two vectors from step to step: the hidden state $\mathbf{h}$, and the **cell state** $\mathbf{c}$. The cell state is its long memory; the hidden state is what it shows of it at each step.

        **Gates.** A **gate** is a vector of numbers between 0 and 1, made with a sigmoid. Multiplied element by element, with $\odot$ as in [Backprop through whole layers by hand](/course/open/nn-backprop-tensors?lang=en), it decides how much of each number goes through: 0 nothing, 1 all of it. An LSTM has three:

        - the **forget gate** $\mathbf{f}_t$: how much of the old cell state stays;
        - the **input gate** $\mathbf{i}_t$: how much of a new **candidate** $\mathbf{g}_t$ goes in;
        - the **output gate** $\mathbf{o}_t$: how much of the cell state the hidden state shows.

        **One step.** As in a plain recurrent network, the input and the hidden state before go through weights: $\mathbf{z}_t = \mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}$, now with four times as many rows. Its four parts are, in PyTorch's order, $\mathbf{i}_t$, $\mathbf{f}_t$, $\mathbf{g}_t$ and $\mathbf{o}_t$: the gates through a sigmoid, the candidate through tanh. Then:
        """),
        L.formula(r"[[hidden:\mathbf{c}_t]] = \mathbf{f}_t \odot [[hidden:\mathbf{c}_{t-1}]] + \mathbf{i}_t \odot \mathbf{g}_t"),
        L.formula(r"[[hidden:\mathbf{h}_t]] = \mathbf{o}_t \odot \tanh([[hidden:\mathbf{c}_t]])"),
        mo.md(r"""
        **Why the gradient survives.** Going back from $\mathbf{c}_t$ to $\mathbf{c}_{t-1}$, the gradient is multiplied by $\mathbf{f}_t$, and that is all: no weight matrix and no slope of tanh, which a plain recurrent network has at every step. Where the forget gate is near 1, the gradient goes back along the cell state almost unchanged, for many steps. In training, the network learns when to keep, $\mathbf{f}_t$ near 1, and when to forget, near 0.

        **The GRU** has no cell state and two gates: a **reset gate** $\mathbf{r}_t$, which decides how much of $\mathbf{h}_{t-1}$ goes into a new candidate $\mathbf{n}_t$, and an **update gate** $\mathbf{z}_t$, which mixes the old hidden state with that candidate. This $\mathbf{z}_t$ is a gate, as PyTorch names it, not the $\mathbf{z}_t$ of the LSTM:
        """),
        L.formula(r"[[hidden:\mathbf{h}_t]] = (1 - \mathbf{z}_t) \odot \mathbf{n}_t + \mathbf{z}_t \odot [[hidden:\mathbf{h}_{t-1}]]"),
        mo.md(r"""
        With three parts instead of four, a GRU has a quarter fewer weights. PyTorch has both: `nn.LSTM` and `nn.GRU` read a whole sequence, `nn.LSTMCell` takes one step.

        The picture: how much each input still changes the last hidden state, against how many steps before the end it was read, measured with PyTorch at the starting weights and averaged over 8 networks with a hidden state of 16 numbers. The y-axis has a log scale: equal steps down are equal factors. PyTorch starts the forget gate around 0.5. The solid line belongs to the same LSTMs, with 3 added to the bias of their forget gate, so that it starts near 0.95. A larger bias for the forget gate at the start is a known trick: Jozefowicz, Zaremba and Sutskever (2015) recommend it.
        """),
        mo.Html(C.reach_picture({"title": "How far back the gradient reaches", "x": "steps back from the end", "y": "share that reaches the end",
                                 "rnn": "recurrent network", "lstm": "LSTM, forget gate 0.5", "keep": "LSTM, forget gate 0.95",
                                 "lang": L.lang,
                                 "alt": "Three lines on a log scale from 10 down to 10 to the minus 16, against 0 to 59 steps back. The plain recurrent network, grey and dashed, falls to about 10 to the minus 16 after 59 steps. The LSTM with its forget gate around 0.5, dashed, falls to about 10 to the minus 12. The LSTM with its forget gate near 0.95, solid, stays between about 0.08 and 1."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **LSTM** | long short-term memory: a recurrent network with a cell state and three gates | `nn.LSTM` |
        | **cell state**, $\mathbf{c}_t$ | the second vector an LSTM carries from step to step; it changes only through the gates | `c` |
        | **gate** | a vector of numbers between 0 and 1, from a sigmoid, that decides element by element how much goes through | |
        | **forget gate**, **input gate**, **output gate**: $\mathbf{f}_t$, $\mathbf{i}_t$, $\mathbf{o}_t$ | how much of the old cell state stays, how much of the candidate goes in, and how much of the cell state the hidden state shows | `f`, `i`, `o` |
        | **candidate**, $\mathbf{g}_t$ | what the step could add to the cell state, from tanh | `g` |
        | **GRU** | gated recurrent unit: no cell state, a reset gate $\mathbf{r}_t$ and an update gate $\mathbf{z}_t$ | `nn.GRU` |
        """),
        L.watch_out("Not every text writes the GRU this way. Some swap $\\mathbf{z}_t$ and $1 - \\mathbf{z}_t$, so that $\\mathbf{z}_t$ goes with the new candidate. The idea is the same; read which one a paper or a library uses before you compare numbers."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One step of an LSTM:

    1. Work out $\mathbf{z}_t = \mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}$ and split it into four equal parts.
    2. Input gate, forget gate and output gate: the sigmoid of their parts; the candidate: the tanh of its part.
    3. New cell state: the forget gate times the old cell state, plus the input gate times the candidate.
    4. New hidden state: the output gate times the tanh of the new cell state.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One step of an LSTM with a cell state of one number, from steps 2 to 4 of the recipe: the four parts of $\mathbf{z}_t$ are given. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_step():
    import math

    _z = [2.0, 4.0, -0.5, 1.0]
    _c = 1.5
    # 2. the gates through a sigmoid, the candidate through tanh
    _i = 1 / (1 + math.exp(-_z[0]))
    _f = 1 / (1 + math.exp(-_z[1]))
    _g = math.tanh(_z[2])
    _o = 1 / (1 + math.exp(-_z[3]))
    print("i, f, g, o:", round(_i, 4), round(_f, 4), round(_g, 4), round(_o, 4))
    # 3. the new cell state
    _c_new = 0.0  # complete: _f * _c + _i * _g
    # 4. the new hidden state
    _h_new = _o * math.tanh(_c_new)
    print("c:", round(_c_new, 4), " h:", round(_h_new, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell state becomes 0, and so does the hidden state: everything is forgotten. Completed, the forget gate, 0.982, keeps almost all of the old 1.5, and the input gate, 0.8808, lets in most of the candidate, −0.4621: the cell state becomes 1.066, and the hidden state $0.7311 \times \tanh(1.066) \approx 0.576$. The candidate is negative, so this step takes some of the cell state away: from 1.5 to 1.066.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One step of an LSTM

        `lstm_step(x, h, c, W_x, W_h, b)` takes one step of an LSTM, with the input `x`, the hidden state `h` and the cell state `c` of the step before, and gives back the new hidden state and the new cell state. `W_x` has shape (4 × hidden, inputs), `W_h` shape (4 × hidden, hidden), and `b` shape (4 × hidden,). The first lines make arrays, work out `z`, and take the input gate and the forget gate from their parts of it. Replace each `...`: the candidate, the output gate, the new cell state and the new hidden state. The check calls `lstm_step([1.0], [0.5], [2.0], [[1.0], [2.0], [0.5], [-1.0]], [[0.5], [1.0], [1.0], [2.0]], [0.0, 1.0, 0.0, 0.5])`, which should give about `([0.6147], [2.5334])`, and tries a hidden state of 2 numbers, weights of 0, and whole numbers.
        """),
        L.task("Make `lstm_step` give the new hidden state and the new cell state of one LSTM step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("lstm-step", lines=18)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("lstm-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "lstm-step",
        [
            "The candidate and the output gate come from the third and the fourth part of `z`, as `i` and `f` come from the first and the second. Then steps 3 and 4 of the recipe, with `*` for $\\odot$.",
            "`g` goes through `np.tanh`, `o` through `sigmoid`. The new hidden state uses the new cell state.",
        ],
        why="`g = np.tanh(z[2 * n:3 * n])`, `o = sigmoid(z[3 * n:4 * n])`, `c_new = f * c + i * g` and `h_new = o * np.tanh(c_new)`. For the example, $\\mathbf{z}$ is (1.25, 3.5, 1, 0.5), so the forget gate is 0.9707 and keeps most of the old 2.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your step and PyTorch's

    The cell below makes PyTorch's `nn.LSTMCell` for inputs of 3 numbers and a hidden state of 2, gives your `lstm_step` the same weights, and compares the new hidden states. PyTorch keeps two bias vectors, one for the input and one for the hidden state; added up, they are your `b`.
    """)
    return


@app.cell(hide_code=True)
def demo_torch(box_step, result_step):
    _both = C.against_torch(box_step.value) if result_step.state == "correct" else None
    mo.md(
        f"Your `lstm_step` gives the hidden state `{_both[0]}`, and `nn.LSTMCell` gives `{_both[1]}`: the same step, in the same order of the gates."
        if _both else "When your `lstm_step` is right, it is compared with PyTorch's `nn.LSTMCell` here."
    )
    return


@app.cell(hide_code=True)
def text_exercise_forget():
    mo.md(r"""
    ### Exercise 2. A forget gate of 0
    """)
    return


@app.cell(hide_code=True)
def choice_forget_cell():
    choice_forget = L.choice(
        "forget-zero",
        question="In a trained LSTM, for one number of the cell state at step $t$, the forget gate is almost 0 and the input gate almost 1. What happens to that number?",
        options={
            "add": "It keeps its old value and adds the candidate",
            "zero": "It becomes 0, and stays 0 from then on",
            "right": "Its old value is replaced by the candidate",
        },
        correct="right",
        why={
            "add": "The forget gate multiplies the old value: almost 0 means almost nothing of it stays.",
            "zero": "The input gate of almost 1 lets the candidate in, and later steps can add to it again.",
            "right": "$0 \\cdot c_{t-1} + 1 \\cdot g_t = g_t$: the old value is gone, and almost no gradient goes back along the cell state past this step. The network has learned to forget here, for example at the end of a sentence.",
        },
    )
    choice_forget
    return (choice_forget,)


@app.cell(hide_code=True)
def text_exercise_gru():
    mo.md(r"""
    ### Exercise 3. A GRU that copies
    """)
    return


@app.cell(hide_code=True)
def choice_gru_cell():
    choice_gru = L.choice(
        "gru-copy",
        question="A GRU, written as in *The idea*, has an update gate $\\mathbf{z}_t$ of 1 for every number at every step. What does its hidden state do?",
        options={
            "right": "It stays as it was: every step copies $\\mathbf{h}_{t-1}$",
            "new": "It forgets everything: every step takes only the candidate",
            "mean": "It becomes the average of the old hidden state and the candidate",
        },
        correct="right",
        why={
            "new": "With $\\mathbf{z}_t = 1$, the candidate gets $1 - 1 = 0$. Read the formula in *The idea* once more.",
            "mean": "That would be $\\mathbf{z}_t = 0.5$. With 1, all weight goes to one of the two.",
            "right": "$(1 - 1) \\odot \\mathbf{n}_t + 1 \\odot \\mathbf{h}_{t-1} = \\mathbf{h}_{t-1}$, and the gradient goes back unchanged. This is how a GRU keeps something for a long time, as an LSTM does with a forget gate of 1.",
        },
    )
    choice_gru
    return (choice_gru,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The step of [Recurrence: a network with memory](/course/open/rnn-memory-idea?lang=en), with gates made by the sigmoid of [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en). Along the cell state, the factors of [A recurrent network and backprop through time](/course/open/rnn-vanilla?lang=en) become forget gates, and those can stay near 1. Transformers take another way to reach far back: every position looks at every other position directly, without going step by step.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_forget, choice_gru, result_step):
    L.wrap_up(result_step, choice_forget, choice_gru)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
