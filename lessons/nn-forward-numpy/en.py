# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-forward-numpy. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-forward-numpy", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en) and [Broadcasting](/course/open/np-broadcasting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="In [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en), `W` has one row of weights per hidden neuron. A hidden layer of 4 neurons gets 3 inputs. What is the shape of its `W`?",
        options={
            "inputs": "(3, 4)",
            "right": "(4, 3)",
            "vector": "(4,)",
        },
        correct="right",
        why={
            "inputs": "That would be one row per input. Which neuron's weights are in `W[i]`? See the recipe in [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).",
            "vector": "That is one number per neuron, like the bias terms `b`. How many weights does each neuron have? See the recipe in [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).",
            "right": "4 rows, one per neuron, and 3 numbers in each row, one weight per input.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_broadcast():
    quick_broadcast = L.choice(
        "quick-broadcast",
        question="`Z` has shape (10, 4) and `b` has shape (4,). What does `Z + b` give?",
        options={
            "error": "NumPy stops: 10 and 4 differ",
            "first": "Only the first row of `Z` gets `b` added",
            "right": "Every row of `Z` gets `b` added, number by number",
        },
        correct="right",
        why={
            "error": "Lined up on the right, 4 and 4 are the same, and the missing length counts as 1. See the rule in [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "first": "NumPy stretches `b` over all the rows. See the picture in [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "right": "From the right, 4 and 4 are the same, and the missing length of `b` counts as 1: `b` is stretched over the 10 rows.",
        },
    )
    quick_broadcast
    return (quick_broadcast,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en) you worked out a network neuron by neuron, for one example at a time, with a loop. A real network has hundreds of neurons in a layer and learns from thousands of examples, and in [Calculating without loops](/course/open/np-vectorization?lang=en) a Python loop was many times slower than NumPy. Without the loop, you calculate with whole matrices, and then one thing needs watching: their shapes. A matrix the wrong way round is the most common mistake in the code of a network.

    **An everyday example.** When you add up amounts in euros and in dollars, you write the currency next to each amount. Then you see at once that you are about to add euros to dollars, before the total is wrong.
    """)
    return


@app.cell(hide_code=True)
def predict_one():
    guess_one = L.guess(
        "one",
        question="A hidden layer of 4 neurons gets 100 examples at once, each with 3 inputs. In [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en), one `np.dot` gave the `z` of one neuron for one example. How many matrix multiplications with `@` do you think the whole layer takes, for all 100 examples?",
        options={
            "each": "400: one for each neuron and each example",
            "examples": "100: one for each example",
            "right": "1",
        },
        reveal="**One.** With the examples as the rows of `X`, shape (100, 3), and the weights as the rows of `W`, shape (4, 3), `X @ W.T` has shape (100, 4). The number in row $k$ and column $j$ is the dot product of example $k$ with the weights of neuron $j$: the $z$ of that neuron for that example. All 400 of them come out of one multiplication.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **forward pass** sends the inputs through a network, from the first layer to the last, to get its outputs. Each layer does the same three things: multiply by its weights, add its bias terms, and put the result through its activation function.

        **One layer, one example.** The weights of a layer form a **weight matrix** $\mathbf{W}$, with one row per neuron, as `W` in [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en): a layer of 4 neurons with 3 inputs has shape (4, 3). `W @ x` takes the dot product of every row with `x`, so it gives the $z$ of every neuron at once, shape (4,). The bias terms of the layer form a vector $\mathbf{b}$ of shape (4,), one per neuron.
        """),
        L.formula(r"[[hidden:\mathbf{h}]] = \mathrm{ReLU}([[param:\mathbf{W}]]\mathbf{x} + [[param:\mathbf{b}]])"),
        mo.md(r"""
        **One layer, all examples.** Put the examples in $\mathbf{X}$, one row per example, as in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en): shape $(n, 3)$. Each $z$ is now the dot product of a row of $\mathbf{X}$ with a row of $\mathbf{W}$. Matrix multiplication takes the rows of the first matrix with the columns of the second, so flip $\mathbf{W}$ first, as in [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en): $\mathbf{W}^\top$ has shape (3, 4), one column per neuron. Then add $\mathbf{b}$: broadcasting adds it to every row. The output $\mathbf{H}$ has one row per example and one column per neuron.
        """),
        L.formula(r"[[hidden:\mathbf{H}]] = \mathrm{ReLU}(\mathbf{X}[[param:\mathbf{W}]]_{[[param:1]]}^\top + [[param:\mathbf{b}_1]])"),
        L.formula(r"[[pred:\hat{\mathbf{Y}}]] = [[hidden:\mathbf{H}]][[param:\mathbf{W}]]_{[[param:2]]}^\top + [[param:\mathbf{b}_2]]"),
        mo.md(r"""
        The small number says which layer: $\mathbf{W}_1$ and $\mathbf{b}_1$ belong to the hidden layer, $\mathbf{W}_2$ and $\mathbf{b}_2$ to the output layer. The output layer does the same with $\mathbf{H}$ as its inputs. With one output neuron, $\mathbf{W}_2$ has one row, and $\hat{\mathbf{Y}}$ one column.

        Write the shape of every step next to it, as a comment: `# (100, 3) @ (3, 4) + (4,) -> (100, 4)`. Before you run the line, you see whether the inner numbers are equal and what comes out, just like the currencies next to the amounts.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **forward pass** | | the inputs sent through a network, layer by layer, from the first to the last, to get its outputs | |
        | $\mathbf{W}$ | capital W | the **weight matrix** of a layer: one row of weights per neuron, shape (neurons, inputs) | `W` |
        | $\mathbf{b}$ | b | the bias terms of a layer, one per neuron, shape (neurons,) | `b` |
        | $\mathbf{W}_1$, $\mathbf{W}_2$ | W one, W two | the weight matrices of layer 1 and layer 2 | `W1`, `W2` |
        | $\mathbf{H}$ | capital H | the outputs of a hidden layer for all examples: one row per example, one column per neuron | `H` |
        | $\hat{\mathbf{Y}}$ | capital Y hat | the outputs of the network for all examples: one row per example, one column per output neuron | `y_pred` |
        | **shape notation** | | the shape of each step written next to it as a comment, so you can follow them from layer to layer | `# (4, 2) @ (2, 3) -> (4, 3)` |
        | `np.maximum` | | the larger of two numbers, number by number: `np.maximum(0, Z)` is ReLU on every number of `Z` at once. Python's `max(0, z)` works on one number only | `np.maximum(0, Z)` |
        """),
        L.watch_out("A square weight matrix, with as many neurons as inputs, fits both ways round. If you forget the `.T`, `X @ W` then runs without an error and gives wrong numbers. So test your code with layers of different sizes, such as 3 inputs and 5 neurons: then a matrix the wrong way round stops NumPy at once."),
        mo.md(r"""
        The hidden layer of the worked example below: 4 examples with 2 inputs, and 3 neurons. The thick boxes show where one number of the result comes from.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_layer():
    mo.vstack([
        draw.array_grids(C.picture(["the second row: example 2", "the third row: neuron 3", "row 2, column 3"]),
                         title="One layer, all examples at once"),
        mo.md(r"""
        The second row of `X`, the example (3, 1), with the third row of `W1`, the weights (1, −1) of neuron 3, plus its bias term 0: 3 × 1 + 1 × (−1) + 0 = 2. That is the number in row 2 and column 3 of the result. Every other number is made the same way, from the row of its example and the row of its neuron.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To work out a forward pass for all examples at once:

    1. Put the examples in `X`, one row per example: shape (examples, inputs).
    2. For each layer, write the shapes next to it and then the line: (examples, inputs) @ (inputs, neurons) + (neurons,) gives (examples, neurons), as `X @ W.T + b`.
    3. A hidden layer puts the result through its activation function: `np.maximum(0, ...)` for ReLU. The output layer here has none.
    4. The output of a layer is the `X` of the next one: its number of columns must equal the number of columns of the next `W`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The network of the picture: 4 examples with 2 inputs, a hidden layer of 3 neurons with ReLU, and an output layer of 1 neuron without an activation function. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_forward():
    import numpy as np

    # 1. four examples with two inputs each
    _X = np.array([[1, 2], [3, 1], [0, 2], [2, 2]])     # (4, 2)
    _W1 = np.array([[1, 1], [0, 1], [1, -1]])           # (3, 2): 3 neurons, 2 inputs
    _b1 = np.array([-2, 0, 0])                          # (3,)
    _W2 = np.array([[1, -1, 2]])                        # (1, 3): 1 neuron, 3 inputs
    _b2 = np.array([0.5])                               # (1,)
    # 2. and 3. the hidden layer: (4, 2) @ (2, 3) + (3,) -> (4, 3), then ReLU
    _H = np.maximum(0, _X @ _W1.T + _b1)
    print(_H.shape)
    print(_H)
    # 4. the output layer, without an activation function: (4, 3) @ (3, 1) + (1,) -> (4, 1)
    _y_pred = _H @ _W2.T + _b2
    print(_y_pred.shape)
    print(_y_pred)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The hidden layer gives one row per example and one column per neuron. For the first example, neuron 3 has $z = 1 - 2 + 0 = -1$, and ReLU makes that 0. The output has shape (4, 1): one row per example, and one column for the one output neuron. For the first example it is $1 \cdot 1 + 2 \cdot (-1) + 0 \cdot 2 + 0.5 = -0.5$. The output layer has no activation function, so a negative output stays negative.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shapes():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The shapes through a network

        A network gets 32 examples at once, with 20 inputs each. Its first hidden layer has 16 neurons, its second 8, and its output layer 3. Replace each `...` with the shape of that array, as a tuple such as `(32, 20)`. `W1`, `W2` and `W3` are the weight matrices of the three layers, `H1` and `H2` the outputs of the two hidden layers, and `y_pred` the output of the network. The last line is a list of the shapes, so the check sees it.
        """),
        L.task("Give the shape of every weight matrix and every output in the network."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shapes():
    box_shapes = L.code_box("shapes", lines=9)
    box_shapes
    return (box_shapes,)


@app.cell(hide_code=True)
def check_shapes(box_shapes):
    result_shapes = L.check("shapes", box_shapes.value)
    result_shapes
    return (result_shapes,)


@app.cell(hide_code=True)
def hints_shapes():
    L.hints(
        "shapes",
        [
            "Go layer by layer, with step 2 of the recipe: a `W` is (neurons, inputs), and its output is (examples, neurons).",
            "The inputs of the second hidden layer are the 16 outputs of the first. The examples stay the rows all the way.",
            "`shape_W1 = (16, 20)` and `shape_H1 = (32, ___)`; then `shape_W2 = (8, ___)`.",
        ],
        why="Each `W` has one row per neuron of its layer and one column per input, and the inputs of a layer are the neurons of the layer before it: (16, 20), (8, 16) and (3, 8). Each output keeps the 32 examples as its rows: (32, 16), (32, 8) and (32, 3).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_forward():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A forward pass for all examples

        Finish `forward(X, W1, b1, W2, b2)`: a network with a hidden layer with ReLU and an output layer without an activation function, for all examples at once. `X` has one row per example: shape (examples, inputs). `W1` has one row of weights per hidden neuron: shape (neurons, inputs); `b1` has one bias term per hidden neuron. `W2` and `b2` are the same for the output layer. The first lines make arrays of what the check passes in. Replace the two `...`: the hidden layer in `H`, and after `return` the output layer. The check calls `forward([[1, 2], [2, 0]], [[1, -1], [0, 1], [2, 1]], [0, -1, -3], [[2, 1, -3]], [0.5])`: that should give `[[-1.5], [1.5]]`. It also tries other networks, with other numbers of examples, inputs and neurons, and more than one output neuron.
        """),
        L.task("Make `forward` work out the hidden layer and the output layer for all examples at once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_forward():
    box_forward = L.code_box("forward", lines=11)
    box_forward
    return (box_forward,)


@app.cell(hide_code=True)
def check_forward(box_forward):
    result_forward = L.check("forward", box_forward.value)
    result_forward
    return (result_forward,)


@app.cell(hide_code=True)
def hints_forward():
    L.hints(
        "forward",
        [
            "Steps 2 and 3 of the recipe, once for the hidden layer and once for the output layer, as in the worked example.",
            "`W1` has one row per neuron, so it needs its `.T` before `@`. ReLU on a whole array is `np.maximum`, not `max`.",
            "`H = np.maximum(0, X @ ___ + b1)` and `return H @ ___ + ___`.",
        ],
        why="For the example `[1, 2]`, the hidden layer has $z = (1 - 2 + 0,\\ 0 + 2 - 1,\\ 2 + 2 - 3) = (-1, 1, 1)$, and ReLU makes that $(0, 1, 1)$. The output is $2 \\cdot 0 + 1 \\cdot 1 - 3 \\cdot 1 + 0.5 = -1.5$. For `[2, 0]`, $\\mathbf{h} = (2, 0, 1)$ and the output is $4 + 0 - 3 + 0.5 = 1.5$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sizes():
    mo.md(r"""
    ### Exercise 3. A test that finds the mistake
    """)
    return


@app.cell(hide_code=True)
def choice_sizes_cell():
    choice_sizes = L.choice(
        "test-sizes",
        question="Your `forward` has a mistake: it says `X @ W1` instead of `X @ W1.T`. With which network does NumPy stop with an error, so that you see the mistake at once?",
        options={
            "square": "2 inputs and a hidden layer of 2 neurons",
            "right": "2 inputs and a hidden layer of 3 neurons",
            "always": "Any network: NumPy always stops when a `.T` is missing",
        },
        correct="right",
        why={
            "square": "Then `W1` has shape (2, 2), and (n, 2) @ (2, 2) fits. What does NumPy do then? See the watch-out in *The idea*.",
            "always": "NumPy only checks whether the inner numbers are equal. When are they equal, even without the `.T`? See the watch-out in *The idea*.",
            "right": "`W1` has shape (3, 2), and in (n, 2) @ (3, 2) the inner numbers 2 and 3 differ: NumPy stops. With as many neurons as inputs, the same mistake would give wrong numbers without an error.",
        },
    )
    choice_sizes
    return (choice_sizes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en) you worked out a network neuron by neuron. Now a whole layer is one line, `X @ W.T + b`, for all examples at once: matrix multiplication, the transpose and broadcasting together. The shape next to each line tells you before you run it whether it fits. In the next lesson, *Too many weights to nudge one by one*, you see why measuring the slope of every weight by nudging it is far too slow for a network like this.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_shapes, result_forward, choice_sizes):
    L.wrap_up(result_shapes, result_forward, choice_sizes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
