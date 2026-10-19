# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-hidden-layer. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-hidden-layer", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What one neuron cannot learn](/course/open/nn-xor?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_why():
    quick_why = L.choice(
        "quick-why",
        question="Why can one neuron with a step not compute XOR?",
        options={
            "right": "Its 1s and 0s lie on two diagonals: no straight line splits those",
            "coarse": "A step is too coarse: with a sigmoid, one neuron could do it",
            "inputs": "It has too few inputs: with a third input, one neuron could do it",
        },
        correct="right",
        why={
            "coarse": "A sigmoid draws the same straight line, at 0.5. See the guess in [What one neuron cannot learn](/course/open/nn-xor?lang=en).",
            "inputs": "XOR has two inputs; the trouble is where its 1s lie. See *The idea* in [What one neuron cannot learn](/course/open/nn-xor?lang=en).",
            "right": "(0, 1) and (1, 0) on one diagonal, (0, 0) and (1, 1) on the other: no straight line splits them.",
        },
    )
    quick_why
    return (quick_why,)


@app.cell(hide_code=True)
def quick_check_rounds():
    quick_rounds = L.choice(
        "quick-rounds",
        question="The perceptron runs on XOR, round after round. What happens?",
        options={
            "settles": "After a few rounds it settles, as on AND",
            "right": "It keeps making mistakes, round after round",
            "small": "It settles as soon as the learning rate is small enough",
        },
        correct="right",
        why={
            "settles": "On AND one straight line splits the classes; on XOR none does. See the worked example in [What one neuron cannot learn](/course/open/nn-xor?lang=en).",
            "small": "A smaller learning rate only makes the steps smaller; no weights without mistakes exist. See [What one neuron cannot learn](/course/open/nn-xor?lang=en).",
            "right": "There are no weights without mistakes, so every round some input is wrong and the line moves again.",
        },
    )
    quick_rounds
    return (quick_rounds,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    One neuron draws one straight line, and no straight line splits XOR. Yet the networks you will train in this course, up to the language model, are built from such neurons, and they learn far more than straight lines. How can that be?

    **An everyday example.** For a cake you whisk the eggs and melt the butter separately, and only then combine them. The two in-between results are not the cake, but the cake needs them.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="Two neurons with a step both see the two inputs. A third neuron with a step sees only their two outputs. Can these three neurons together compute XOR?",
        options={
            "no": "No: each neuron still draws only one straight line",
            "right": "Yes: the third neuron can combine the lines of the other two",
            "inputs": "Only if the third neuron also sees the inputs",
        },
        reveal="**Yes.** The first neuron fires when at least one input is 1, the second when both are. The third fires when the first does and the second does not: for exactly one 1. Each neuron still draws one straight line, but the third one draws its line through the outputs of the other two, and seen from the inputs, that cuts out a band between two lines.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **layer** is a group of neurons that all get the same inputs, each with its own weights and bias term. A **hidden layer** sits between the inputs and the output: you never see its outputs $\mathbf{h}$ directly, only what the neurons after it do with them. Layers one after another make a **multilayer perceptron**, or **MLP**, also when its neurons use an activation function other than a step.

        XOR takes three neurons with a step. Hidden neuron 1 fires when at least one input is 1, hidden neuron 2 when both are, and the output neuron fires when the first does and the second does not, with the weights 1 and −1 and the bias term −0.5 of the neuron that fired for one corner in [A neuron](/course/open/nn-neuron?lang=en):
        """),
        L.formula(r"\begin{aligned} h_1 &= \mathrm{step}(x_1 + x_2 - 0.5) \\ h_2 &= \mathrm{step}(x_1 + x_2 - 1.5) \\ [[pred:\hat{y}]] &= \mathrm{step}(h_1 - h_2 - 0.5) \end{aligned}"),
        mo.md(r"""
        Each hidden neuron draws a line; the output neuron fires between them. But the hidden neurons need an activation function that is not a straight line itself. Without one, hidden neuron $i$ passes on its weighted sum $z = \mathbf{w}_i \cdot \mathbf{x} + b_i$, and the output neuron, with the weights $v_1$ and $v_2$ and the bias term $c$, adds those up: that is again one weighted sum of the inputs plus one bias term, one straight line.
        """),
        L.formula(r"\begin{aligned} & v_1(\mathbf{w}_1 \cdot \mathbf{x} + b_1) + v_2(\mathbf{w}_2 \cdot \mathbf{x} + b_2) + c \\ ={} & (v_1\mathbf{w}_1 + v_2\mathbf{w}_2) \cdot \mathbf{x} + (v_1 b_1 + v_2 b_2 + c) \end{aligned}"),
        mo.md(r"""
        A step is not a straight line, but it has a flaw: it is flat everywhere but at 0, so gradient descent gets no slope to follow. Hidden layers usually use **ReLU** instead: $\mathrm{ReLU}(z) = \max(0, z)$. Below 0 it gives 0, above 0 it passes $z$ on as it is. That bend at 0 is enough to make the network more than one straight line, and above 0 it has a slope of 1. Below 0 ReLU is flat too, so a neuron whose $z$ stays below 0 gets no slope and stays stuck at 0. **Leaky ReLU** lets a little through below 0, such as $0.01 \cdot z$, so that this cannot happen.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **layer** | | neurons that all get the same inputs, each with its own weights and bias term | |
        | **hidden layer** | | a layer between the inputs and the output, whose outputs you only see through the neurons after it | |
        | $\mathbf{h}$ | h | the outputs of a hidden layer, one per hidden neuron | `h` |
        | **ReLU** | rel-oo | the activation function $\max(0, z)$: 0 below 0, $z$ itself above it | `max(0, z)` |
        | **leaky ReLU** | | like ReLU, but with a small slope below 0, such as $0.01 \cdot z$ | |
        | **MLP**, multilayer perceptron | em-el-pee | layers one after another, with an activation function after each hidden layer | |
        """),
        L.watch_out("Count the layers with weights. The XOR network has two: the hidden layer and the output neuron. The inputs are not a layer of neurons; they have no weights of their own."),
        mo.md(r"""
        Move the slider: it sets the bias term $b_2$ of hidden neuron 2. The <span class="c-pred">blue dots</span> show where the network fires; the dashed lines are the two hidden neurons. The corners are XOR: filled for 1, open for 0. A <span class="c-loss">purple ring</span> marks each corner the network gets wrong, and the other chart counts them for every $b_2$; drag across it to set $b_2$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b2_cell():
    def _frame(b2):
        b2 = round(b2, 1)
        top = 0.0 - b2   # hidden neuron 2 fires where x₁ + x₂ lies above this; 0.0 - keeps −0.0 out of the text
        out = C.outputs(b2)
        wrong = C.wrong(b2)
        ringed = ", marked with a purple ring: " + " and ".join(str(c) for c in C.wrong_corners(b2)) if wrong else ""
        if top >= 2.5:   # the second line only touches the top right corner of the plane
            where = "the network fires everywhere above the first dashed line; the second only touches the top right corner of the plane"
        elif top > 0.5:
            where = f"the network fires where x₁ + x₂ lies above 0.5 but not above {L.number(top, 1)}, between the two dashed lines"
        else:
            where = "the second dashed line lies on or below the first, and the network fires nowhere"
        return (C.picture(b2, {"title": "Where the network fires", "wrong_title": "Corners wrong, for every b₂", "wrong": "corners",
                               "alt": f"The plane of the two inputs with the corners of XOR. With b₂ = {L.number(b2, 1)}, {where}; it gets {wrong} of the 4 corners wrong{ringed}. A second chart counts the corners wrong for every b₂: one up to −2, none up to −1, and two above it."}),
                f"""With <i>b</i>₂ = {L.number(b2, 1)}, the network gives {", ".join(str(o) for o in out)} for the four corners, and gets <span class="c-loss ea-num">{wrong}</span> of them wrong.""")

    explore_b2 = L.explore("b2", "b₂", _frame, start=-1.5, low=-2.5, high=0, step=0.1, role="param", task="For which $b_2$ does the network compute XOR? And which corners does it get wrong below and above that?")
    explore_b2
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With $b_2$ above −2 and up to −1, the band holds (0, 1) and (1, 0) and no other corner: XOR. At −2 and below, hidden neuron 2 never fires for a corner, and the network is OR: it gets (1, 1) wrong. Above −1 the band is too narrow to hold a corner, and the network gets (0, 1) and (1, 0) wrong; from −0.5 on the band is gone.

    ### Recipe

    To work out what a network with one hidden layer gives for the inputs `x`, with the weights `W` of the hidden layer, one row per hidden neuron, its bias terms `b`, and the weights `v` and the bias term `c` of the output neuron:

    1. For each hidden neuron `i`: `z = np.dot(W[i], x) + b[i]`, then its activation function, ReLU or a step, gives `h[i]`.
    2. The output neuron sees `h` as its inputs: `np.dot(v, h) + c`, with its own activation function or none.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The XOR network of *The idea*, with a step in all three neurons, on all four inputs. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_xor():
    import numpy as np

    def _step(z):
        if z > 0:
            return 1
        return 0

    _W = [[1, 1], [1, 1]]   # one row of weights per hidden neuron
    _b = [-0.5, -1.5]
    _v = [1, -1]
    _c = -0.5
    for _x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        # 1. the hidden layer
        _h = []
        for _i in range(2):
            _h.append(_step(np.dot(_W[_i], _x) + _b[_i]))
        # 2. the output neuron
        print(_x, _h, _step(np.dot(_v, _h) + _c))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    For (0, 0) neither hidden neuron fires. For (0, 1) and (1, 0) only the first does, and the output neuron gets $1 - 0 - 0.5 = 0.5$: it fires. For (1, 1) both fire, and it gets $1 - 1 - 0.5 = -0.5$: it does not. The hidden layer has turned the inputs into new features, $h_1$ and $h_2$, and on those, XOR is one straight line.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_collapse():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A hidden layer without an activation function
    """)
    return


@app.cell(hide_code=True)
def choice_collapse_cell():
    choice_collapse = L.choice(
        "collapse",
        question="A network has a hidden layer of 100 neurons without an activation function, and an output neuron without one. What can it learn?",
        options={
            "lines": "Much more than one neuron: its 100 hidden neurons draw 100 different lines, and the output neuron combines them",
            "nothing": "Nothing at all: without an activation function the hidden layer gives 0",
            "right": "Only what one neuron without an activation function can: one weighted sum of the inputs plus a bias term",
        },
        correct="right",
        why={
            "lines": "The lines only bend the boundary if something non-linear comes after them. What is a weighted sum of weighted sums? See the second formula in *The idea*.",
            "nothing": "Without an activation function a neuron passes on $z$ itself, not 0. What does the output neuron make of those? See the second formula in *The idea*.",
            "right": "A weighted sum of weighted sums is one weighted sum, with one bias term: the 100 neurons collapse into one. Only an activation function such as ReLU keeps them apart.",
        },
    )
    choice_collapse
    return (choice_collapse,)


@app.cell(hide_code=True)
def text_exercise_forward():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Through the hidden layer

        Finish `forward(x, W, b, v, c)`: a network with a hidden layer with ReLU and an output neuron without an activation function. `W` has one row of weights per hidden neuron and `b` one bias term per hidden neuron; `v` and `c` are the weights and the bias term of the output neuron. Replace the `...` with step 1 of the recipe for hidden neuron `i`, with ReLU, so that its output is added to `h`. The check calls `forward([1, 2], [[1, -1], [0.5, 0.5]], [0, -1], [2, 1], 0.5)`: that should give `1.0`. It also tries other inputs and networks, with more inputs and more hidden neurons.
        """),
        L.task("Make `forward` work out each hidden neuron with ReLU and add its output to `h`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_forward():
    box_forward = L.code_box("forward", lines=9)
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
            "Step 1 of the recipe, as in the worked example, with ReLU instead of the step.",
            "Hidden neuron `i` has the weights `W[i]` and the bias term `b[i]`. What does ReLU do with a `z` below 0, and with one above 0?",
            "`z = np.dot(___, x) + ___`, then `h.append(max(___, z))`.",
        ],
        why="For `[1, 2]`, the first hidden neuron (`i = 0`) has $z = 1 - 2 + 0 = -1$, and ReLU makes that 0. The second (`i = 1`) has $z = 0.5 + 1 - 1 = 0.5$, which ReLU passes on. The output is $2 \\cdot 0 + 1 \\cdot 0.5 + 0.5 = 1$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_output():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. XOR with ReLU

        Two hidden neurons with ReLU: $h_1 = \mathrm{ReLU}(x_1 + x_2)$ and $h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)$. The output neuron has no activation function: $\hat{y} = v_1 h_1 + v_2 h_2 + c$. Choose its weights `v` and its bias term `c`, so that the network gives 0, 1, 1 and 0 for (0, 0), (0, 1), (1, 0) and (1, 1): XOR. Replace the two `...`, and leave the loop as it is. Want to try a few? Do that in the scratchpad.
        """),
        L.task("Choose `v` and `c` so that the network computes XOR."),
    ])
    return


@app.cell(hide_code=True)
def exercise_output():
    box_output = L.code_box("output-weights", lines=14)
    box_output
    return (box_output,)


@app.cell(hide_code=True)
def check_output(box_output):
    result_output = L.check("output-weights", box_output.value)
    result_output
    return (result_output,)


@app.cell(hide_code=True)
def hints_output():
    L.hints(
        "output-weights",
        [
            "First work out $h_1$ and $h_2$ for each corner. Then: which output does XOR want there?",
            "(0, 0) gives $h = (0, 0)$, so the output there is just $c$. (0, 1) and (1, 0) give $(1, 0)$, and (1, 1) gives $(2, 1)$.",
            "$c = 0$ and $v_1 = 1$ follow from the first three corners. Then (1, 1): $2 + v_2 = 0$, so `v = [1, ___]`.",
        ],
        why="With `v = [1, -2]` and `c = 0`: (0, 0) gives 0, (0, 1) and (1, 0) give 1, and (1, 1) gives $2 - 2 \\cdot 1 = 0$. The first hidden neuron counts the ones; the second only starts above one 1, and the output takes it away twice.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What one neuron cannot learn](/course/open/nn-xor?lang=en) one straight line was not enough for XOR. A hidden layer with a non-linear activation function, a step or ReLU, gives the output neuron new features, on which XOR is one straight line. In the next lesson, *The forward pass in NumPy, with shapes*, you work out a whole layer at once with a matrix, and follow the shapes from layer to layer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_collapse, result_forward, result_output):
    L.wrap_up(choice_collapse, result_forward, result_output)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
