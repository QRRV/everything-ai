# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rnn-memory-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rnn-memory-idea", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en) and [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`W` has shape (3, 2), `x` has shape (2,), and `b` has shape (3,). What is the shape of `W @ x + b`?",
        options={
            "inputs": "(2,): one number per input",
            "matrix": "(3, 2), the same as `W`",
            "right": "(3,): one number per row of `W`",
        },
        correct="right",
        why={
            "inputs": "`W @ x` takes the dot product of every row of `W` with `x`. See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "matrix": "A matrix times a vector gives a vector: one dot product per row. See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "right": "(3, 2) @ (2,) -> (3,): one dot product for each of the 3 rows of `W`, and `b` adds one number to each.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="Do `[ord(ch) for ch in \"ab\"]` and `[ord(ch) for ch in \"ba\"]` give the same list?",
        options={
            "right": "No: the same numbers, in another order",
            "same": "Yes: both texts have an a and a b",
            "other": "No: the b gets another number when it comes first",
        },
        correct="right",
        why={
            "same": "A list keeps its order: `[97, 98]` is not `[98, 97]`. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "other": "`ord` gives every character its own number, wherever it stands. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "right": "`[97, 98]` against `[98, 97]`: a text is a sequence, and its order is part of it.",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A layer as in [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en) takes a fixed number of inputs: its `W` has one column per input. A text is not like that. One sentence has 12 characters, the next 80, and what a word means depends on what came before it: "not good" is not "good". A network for text has to read a sequence of any length, in order, and keep in mind what it has read so far.

    **An everyday example.** You follow spoken directions through a town: left, two blocks straight on, right, and so on. You only keep where you are and which way you face, and change that at every new instruction. Ten instructions or a hundred, it is the same few things in your head, and the same instructions in another order take you somewhere else.
    """)
    return


@app.cell(hide_code=True)
def predict_trace():
    guess_trace = L.guess(
        "trace",
        question="A network reads a text one character at a time and keeps a short summary of what it has read: two numbers, made again at every character from the new character and the summary before. It reads one \"a\" and then twenty \"b\"s. What do you think is left of the \"a\" in the summary at the end?",
        options={
            "all": "All of it: what is in the summary stays there",
            "more": "More than at the start: it has been carried for longer",
            "right": "Almost nothing",
        },
        reveal=f"**Almost nothing.** With the network of the worked example below, the trace of the a halves at every step: {L.number(C.trace(0), 2)}, {L.number(C.trace(1), 2)}, {L.number(C.trace(2), 2)}, {L.number(C.trace(3), 2)}, … After twenty b's, less than a millionth is left. Every step makes a new summary from the new character and only part of the old one. A trained network can learn weights that keep things longer, but a summary of a few numbers cannot hold everything.",
    )
    guess_trace
    return (guess_trace,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **recurrent neural network**, an RNN, reads a sequence one step at a time, such as a text character by character. It carries a **hidden state** $\mathbf{h}$ from step to step: a vector of numbers that sums up what it has read so far. At step $t$ it reads the input $\mathbf{x}_t$, and makes the new hidden state from that input and the hidden state of the step before:
        """),
        L.formula(r"[[hidden:\mathbf{h}_t]] = \tanh([[param:\mathbf{W}_x]] [[data:\mathbf{x}_t]] + [[param:\mathbf{W}_h]] [[hidden:\mathbf{h}_{t-1}]] + [[param:\mathbf{b}]])"),
        mo.md(r"""
        - The hidden state starts at zeros: $\mathbf{h}_0 = \mathbf{0}$.
        - $\mathbf{W}_x$ weighs the new input and $\mathbf{W}_h$ the hidden state before, as the `W` of a layer: one row for each number of the hidden state. With a hidden state of 2 numbers and inputs of 2 numbers, both are (2, 2).
        - The same $\mathbf{W}_x$, $\mathbf{W}_h$ and $\mathbf{b}$ at every step. So the network reads a sequence of any length with the same weights, and the hidden state keeps the same size at every step.
        - **tanh** squeezes every number between −1 and 1, with 0 in the middle, as the sigmoid of [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) squeezes between 0 and 1. So the hidden state cannot grow step after step.

        The last hidden state sums up the whole sequence. An output layer can read it, for example to say whether a review is positive or negative. In a network that learns, training finds $\mathbf{W}_x$, $\mathbf{W}_h$ and $\mathbf{b}$; in this lesson you set them by hand, so you can see what the hidden state does.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **recurrent neural network** | RNN, R N N | a network that reads a sequence one step at a time, and passes a hidden state from each step to the next, with the same weights at every step | |
        | **hidden state** | | the vector a recurrent network carries from step to step: a summary of what it has read so far | `h` |
        | $\mathbf{x}_t$ | x t | the input at step $t$ | `x` |
        | $\mathbf{h}_t$, $\mathbf{h}_{t-1}$ | h t, h t minus 1 | the hidden state after step $t$, and after the step before | `h` |
        | $\mathbf{W}_x$, $\mathbf{W}_h$ | W x, W h | the weights for the input, and for the hidden state before | `W_x`, `W_h` |
        | **tanh** | tanh | squeezes any number between −1 and 1, with 0 in the middle | `np.tanh(z)` |
        """),
        L.watch_out("*State* means different things in AI. Here it is the **hidden state**, the vector a recurrent network carries from step to step: always say hidden state in full. In reinforcement learning, a *state* is the situation a model is in when it picks its next action, which is something else."),
        mo.md(r"""
        The picture: the network of the worked example reads "abbb". After every step, the bars show the two numbers of the hidden state. The first one remembers the a's, the second the b's.
        """),
        mo.Html(C.picture("abbb", {"title": "One network, four steps", "step": "step",
                                   "alt": "Five boxes from left to right, h0 to h4, joined by arrows marked W h; below boxes 1 to 4 the characters a, b, b, b, each with an arrow W x up. In each box two bars, a and b. h0: both 0. h1: a 0.76, b 0. h2: a 0.36, b 0.76. h3: a 0.18, b 0.88. h4: a 0.09, b 0.89. The bar of the a gets shorter at every step."}, L.lang)),
        mo.md(r"""
        The a enters at step 1 and then fades: every step keeps only part of it. The b's push the second number up, but tanh keeps it below 1.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a text with a recurrent network:

    1. Choose a vector for every character, the weights `W_x` and `W_h`, and the bias terms `b`: the same at every step.
    2. Start with a hidden state of zeros: `h = np.zeros(size)`.
    3. For every character, in order: `h = np.tanh(W_x @ x + W_h @ h + b)`, with `x` the vector of that character.
    4. Read the last hidden state: it sums up the whole text.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The network of the picture, with a hidden state of 2 numbers, reads three texts made of a's and b's. The dictionary `vec` gives every character its vector, as in [Dictionaries](/course/open/py-dicts?lang=en). The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_rnn():
    import numpy as np

    # 1. a vector for every character, the weights and the bias terms: the same at every step
    _vec = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    _W_x = np.array([[1.0, 0.0], [0.0, 1.0]])
    _W_h = np.array([[0.5, 0.0], [0.0, 0.5]])
    _b = np.array([0.0, 0.0])

    for _text in ["ab", "ba", "abbbb"]:
        # 2. the hidden state starts at zeros
        _h = np.zeros(2)
        # 3. one step for every character, in order
        for _ch in _text:
            _h = np.tanh(_W_x @ _vec[_ch] + _W_h @ _h + _b)   # (2, 2) @ (2,) + (2, 2) @ (2,) + (2,) -> (2,)
        # 4. the last hidden state sums up the text
        print(_text, _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    _ab, _ba, _long = C.states("ab")[-1], C.states("ba")[-1], C.states("abbbb")[-1]
    mo.md(f"""
    `W_x` puts a new a in the first number and a new b in the second, and `W_h` keeps half of what was there. "ab" and "ba" have the same characters, but their hidden states differ: ({L.number(_ab[0], 2)}, {L.number(_ab[1], 2)}) against ({L.number(_ba[0], 2)}, {L.number(_ba[1], 2)}). The last character counts most, so the order is in the hidden state. In "abbbb", the a halves at every step and is down to {L.number(_long[0], 2)}. The four b's do not push the second number past 1: it is {L.number(_long[1], 2)}, because tanh keeps it below 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_size():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How big is the memory?
    """)
    return


@app.cell(hide_code=True)
def choice_size_cell():
    choice_size = L.choice(
        "size",
        question="A recurrent network with a hidden state of 8 numbers reads a text of 1,000 characters. How many numbers does its hidden state hold after the last character?",
        options={
            "per-char": "8,000: 8 for every character",
            "chars": "1,000: one for every character",
            "right": "8, as after the first character",
        },
        correct="right",
        why={
            "per-char": "Every step makes a new hidden state of the same size, and it replaces the old one. See *The idea*.",
            "chars": "The hidden state has as many numbers as `W_h` has rows, whatever the length of the text. See *The idea*.",
            "right": "Every step replaces the hidden state with a new one of the same size: 8 numbers have to sum up 1,000 characters.",
        },
    )
    choice_size
    return (choice_size,)


@app.cell(hide_code=True)
def text_exercise_rnn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A recurrent network for any sequence

        `rnn(xs, W_x, W_h, b)` reads the inputs in `xs` in order, one vector per step, and gives back the last hidden state. `W_x` has shape (hidden, inputs), `W_h` shape (hidden, hidden), and `b`, the bias terms, shape (hidden,): the hidden state has as many numbers as `b`. The first line makes arrays of them. Replace the two `...`: the hidden state to start with, and one step of the network. The check calls `rnn([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]], [0, 0])`, the "ab" of the worked example, and that should give about `[0.3634, 0.7616]`. It also tries hidden states of other sizes, and longer sequences.
        """),
        L.task("Make `rnn` start with a hidden state of zeros and take one step for every input."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rnn():
    box_rnn = L.code_box("rnn", lines=10)
    box_rnn
    return (box_rnn,)


@app.cell(hide_code=True)
def check_rnn(box_rnn):
    result_rnn = L.check("rnn", box_rnn.value)
    result_rnn
    return (result_rnn,)


@app.cell(hide_code=True)
def hints_rnn():
    L.hints(
        "rnn",
        [
            "Steps 2 and 3 of the recipe, with the names of the function.",
            "`h` starts with one zero for every number of the hidden state: `b` has that many. Every step uses the input `x`, the hidden state `h` of the step before, both weight matrices and the bias terms.",
            "`h = np.zeros(len(___))` before the loop, and `h = np.tanh(W_x @ x + W_h @ ___ + b)` in it.",
        ],
        why="With the example, step 1 reads (1, 0): `W_x @ x` is (1, 0), and `W_h @ h` and `b` are zeros, so tanh of (1, 0) gives (0.7616, 0). Step 2 reads (0, 1), and `W_h @ h` adds half of the hidden state before: tanh of (0.3808, 1) gives (0.3634, 0.7616).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_memory():
    mo.md(r"""
    ### Exercise 3. No memory
    """)
    return


@app.cell(hide_code=True)
def choice_memory_cell():
    choice_memory = L.choice(
        "no-memory",
        question="In a recurrent network you set every weight of `W_h` to 0, and keep `W_x` and `b`. It reads a text. What does the last hidden state depend on now?",
        options={
            "right": "Only on the last character",
            "all": "On all characters, each as much as the others",
            "nothing": "On nothing: it stays all zeros",
        },
        correct="right",
        why={
            "all": "`W_h @ h` is how the hidden state before gets into a step. With `W_h` all zeros, that part is zeros. See the formula in *The idea*.",
            "nothing": "`W_x @ x + b` still brings in the new character at every step. See the formula in *The idea*.",
            "right": "Every step is then `np.tanh(W_x @ x + b)`: the hidden state before is multiplied by 0. Each step forgets everything, and the last hidden state is made from the last character alone.",
        },
    )
    choice_memory
    return (choice_memory,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A recurrent network does at every step what a layer of [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en) does: weights times inputs, plus bias terms, through an activation function. What is new is the input it gets from itself: the hidden state of the step before. That way it reads a text of [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en) in order, at any length. Its memory has a fixed size and fades, so the start of a long text gets lost. In the next lesson, *Attention: every word looks at the others*, every position looks straight at every other one, however far back it is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_memory, choice_size, result_rnn):
    L.wrap_up(choice_size, result_rnn, choice_memory)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
