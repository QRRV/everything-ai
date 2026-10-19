# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-why-backprop. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-why-backprop", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two parameters: the gradient](/course/open/math-gradient?lang=en) and [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="You move `w` by a tiny step `h` = 0.001, and the loss goes from 50 to 49.997. About what is the slope of the loss for `w`?",
        options={
            "right": "−3",
            "change": "−0.003",
            "sign": "3",
        },
        correct="right",
        why={
            "change": "That is the change of the loss. A slope is the change per step: what do you divide it by? See the recipe in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "sign": "The loss goes down when `w` goes up. What does that say about the sign of the slope? See [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "right": "The change of the loss, −0.003, divided by the step 0.001: −3. The loss goes down when `w` goes up.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="A layer has 784 inputs and 128 neurons. How many weights does its `W` hold?",
        options={
            "sum": "912",
            "right": "100,352",
            "neurons": "128",
        },
        correct="right",
        why={
            "sum": "That is 784 + 128. Each neuron has a weight for each input: how many is that together? See the shapes in [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "neurons": "That is one number per neuron, like the bias terms. `W` has shape (128, 784): how many numbers is that? See [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en).",
            "right": "`W` has shape (128, 784): 128 rows of 784 weights, 128 × 784 = 100,352.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Training takes a step against the gradient, again and again, and every step needs the slope of the loss for every parameter. In [Two parameters: the gradient](/course/open/math-gradient?lang=en) you measured those slopes by moving each parameter in turn by a tiny step, for $w$ and $b$. A small network for handwritten digits already has about 100,000 parameters, and a language model has billions. Before you build anything that large, it pays to count what it costs.

    **An everyday example.** At a party where everyone shakes hands with everyone, 4 guests make 6 handshakes. With 40 guests it is not 60 but 780: ten times as many guests, more than a hundred times as many handshakes.
    """)
    return


@app.cell(hide_code=True)
def predict_seconds():
    guess_seconds = L.guess(
        "seconds",
        question="Say one forward pass of a network with 100,000 parameters takes 0.1 milliseconds, for one example. How long do you think measuring the gradient takes for that one example, by moving each parameter in turn?",
        options={
            "twice": "About 0.2 milliseconds: once where you are, once moved",
            "right": "About 10 seconds",
            "second": "About 1 second",
        },
        reveal="**About 10 seconds.** You need the loss once where you are, and once more for each parameter you move: 100,001 forward passes. At 0.1 milliseconds each, that is about 10 seconds, for one example and one step. Training takes thousands of steps, each over many examples.",
    )
    guess_seconds
    return (guess_seconds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        To compare ways of calculating, you count their **FLOPs**: the number of floating-point operations, additions and multiplications, that they need. That number does not depend on the computer.

        **One forward pass.** In a layer, each weight is used once per example: one multiplication, $w \cdot x$, and one addition to the sum (the first one adds the bias term). With $N$ parameters, a forward pass for one example costs about $2N$ FLOPs.

        **Measuring by moving.** You work out the loss where you are, and then once more for each parameter, moved by a tiny step: $N + 1$ forward passes. That is $N + 1$ times the cost of one:
        """),
        L.formula(r"\underbrace{2N}_{\text{one forward pass}} \qquad \underbrace{(N + 1) \cdot 2N \approx 2N^2}_{\text{every slope by moving}}"),
        mo.md(r"""
        What matters is how the cost grows with $N$. **Big-O notation** says just that, and leaves out the constant factors: a forward pass is $O(N)$, it grows like $N$; measuring every slope by moving is $O(N^2)$, it grows like $N^2$. Make the network twice as big, and a forward pass takes twice as long, but measuring the gradient four times as long.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **FLOPs** | flops | the number of floating-point operations, such as additions and multiplications, that a calculation needs: a way to count its cost | |
        | $N$ | N | the number of parameters of a model: all its weights and bias terms; not $n$, the number of examples | `n_params` |
        | **big-O notation** | | how the cost grows with the size, without the constant factors | |
        | $O(N)$ | big O of N | grows like $N$: twice as big, twice the cost | |
        | $O(N^2)$ | big O of N squared | grows like $N^2$: twice as big, four times the cost | |
        """),
        L.watch_out("Measuring slopes by moving is not wrong, only slow. It is simple and hard to get wrong, so it stays the way to check slopes that were worked out in a faster way, on a few parameters."),
        mo.md(r"""
        Move the slider, or drag across the chart, to set the number of parameters $N$. The dashed line is one forward pass, the solid line measuring every slope by moving, both in millions of FLOPs.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_n_cell():
    def _frame(n):
        n = int(n)
        return (C.picture(n, {"title": "What the slopes cost", "x": "number of parameters N", "y": "millions of FLOPs",
                              "move": "every slope by moving", "forward": "one forward pass",
                              "alt": f"Two lines over N from 0 to 1000. The dashed line, one forward pass, stays near 0: at N = 1000 it is 0.002 million FLOPs. The solid line, every slope by moving, curves up to about 2 million FLOPs at N = 1000. At N = {n}, the dots are at {L.number(C.forward(n) / 1e6, 4)} and {L.number(C.moving(n) / 1e6, 3)} million."}),
                f"""With <i>N</i> = {n}, one forward pass costs <span class="ea-num">{C.whole(C.forward(n), L.lang)}</span> FLOPs, and every slope by moving <span class="ea-num">{C.whole(C.moving(n), L.lang)}</span>: {C.whole(n + 1, L.lang)} times as many.""")

    explore_n = L.explore("n", "N", _frame, start=400, low=50, high=1000, step=50, role="", task="Say you have a million FLOPs for the slopes of one example. Up to which $N$ is that enough to measure every slope by moving?")
    explore_n
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    A million FLOPs is enough to measure every slope by moving up to about $N$ = 700: that takes 981,400 FLOPs, and at 750 it is already 1,126,500. For one forward pass, the same million is enough up to $N$ = 500,000.

    ### Recipe

    To estimate what a calculation costs:

    1. Count the FLOPs of one step, such as one forward pass: about 2 per weight.
    2. Count how often the step is done, such as one forward pass for each parameter, plus one.
    3. Multiply the two, and keep only the part that grows fastest: that is its big-O.
    4. Fill in the size you have, and see whether it is still doable.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A network for handwritten digits: 784 inputs, a hidden layer of 128 neurons, and 10 outputs. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_cost():
    _weights = 128 * 784 + 10 * 128     # W1 has shape (128, 784), W2 (10, 128)
    _n_params = _weights + 128 + 10     # plus b1 and b2
    # 1. one forward pass, for one example: 2 FLOPs per weight
    _forward = 2 * _weights
    print(_n_params, _forward)
    # 2. and 3. one forward pass where you are, and one for each parameter
    _moving = (_n_params + 1) * _forward
    print(_moving)
    # 4. how many times as much as one forward pass
    print(_moving / _forward)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The network has 101,770 parameters, and one forward pass for one example costs 203,264 FLOPs. Measuring every slope by moving costs about 20,700,000,000 FLOPs: 101,771 times as much, for one example. Training needs a way to get every slope at a few times the cost of a forward pass.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flops():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The FLOPs of a forward pass

        `sizes` lists the number of inputs and then the number of neurons in each layer: `[784, 128, 10]` is the network of the worked example. Finish `forward_flops(sizes)`: it gives the FLOPs of one forward pass for one example, 2 per weight. Replace the `...` with what layer `i + 1` adds to `total`. The check calls `forward_flops([3, 4, 2])`: that should give `40`. It also tries other networks, with more and fewer layers.
        """),
        L.task("Make `forward_flops` count 2 FLOPs for each weight of the network."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flops():
    box_flops = L.code_box("forward-flops", lines=6)
    box_flops
    return (box_flops,)


@app.cell(hide_code=True)
def check_flops(box_flops):
    result_flops = L.check("forward-flops", box_flops.value)
    result_flops
    return (result_flops,)


@app.cell(hide_code=True)
def hints_flops():
    L.hints(
        "forward-flops",
        [
            "Step 1 of the recipe, as in the worked example, but for any list of sizes.",
            "Layer `i + 1` has `sizes[i + 1]` neurons, each with a weight for each of its `sizes[i]` inputs.",
            "`total = total + 2 * ___ * ___`.",
        ],
        why="For `[3, 4, 2]`, the hidden layer has 4 × 3 = 12 weights and the output layer 2 × 4 = 8: 20 weights, so 40 FLOPs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_growth():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Like N or like N²?

        Each of these grows with the number of parameters $N$ of a network. Put each in its group.
        """),
        L.task("Sort the six calculations by how they grow with N."),
    ])
    return


@app.cell(hide_code=True)
def sort_growth_cell():
    sort_growth = L.sort("growth")
    sort_growth
    return (sort_growth,)


@app.cell(hide_code=True)
def check_growth(sort_growth):
    result_growth = L.check("growth", sort_growth)
    result_growth
    return (result_growth,)


@app.cell(hide_code=True)
def hints_growth():
    L.hints(
        "growth",
        [
            "For each one: how many forward passes, or how many steps, does it take, and what does each cost?",
            "A fixed bit of work per parameter is $O(N)$. A forward pass for each parameter, or a step for each pair of parameters, is $O(N^2)$.",
            "Like $N$: one forward pass, \\_\\_\\_ and saving the parameters. Like $N^2$: every slope by moving, \\_\\_\\_ and comparing every pair.",
        ],
        why="**Like $N$:** one forward pass, saving the parameters and one step of gradient descent; each does a fixed bit of work per parameter, so twice as many parameters is twice the work. **Like $N^2$:** every slope by moving, half of the slopes by moving and comparing every pair. Moving only half the parameters is still a forward pass, $2N$ FLOPs, for each of $N/2$ parameters: $N^2$, and big-O leaves out the half.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bigger():
    mo.md(r"""
    ### Exercise 3. Ten times as big
    """)
    return


@app.cell(hide_code=True)
def choice_bigger_cell():
    choice_bigger = L.choice(
        "bigger",
        question="For a network with 100,000 parameters, measuring every slope by moving takes about 10 seconds per example. You make the network ten times as big: 1,000,000 parameters. About how long does it take now?",
        options={
            "ten": "About 100 seconds",
            "twenty": "About 20 seconds",
            "right": "About 1,000 seconds",
        },
        correct="right",
        why={
            "ten": "There are ten times as many forward passes. But what happens to the cost of each forward pass? See *The idea*.",
            "twenty": "The cost does not grow by a step of 10 seconds. How does $O(N^2)$ grow when $N$ is ten times as big? See *The idea*.",
            "right": "Ten times as many forward passes, each ten times as expensive: a hundred times as long, about 17 minutes, for one example.",
        },
    )
    choice_bigger
    return (choice_bigger,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two parameters: the gradient](/course/open/math-gradient?lang=en) you measured every slope by moving each parameter in turn. That costs a forward pass per parameter: $O(N^2)$ FLOPs, far too much for a network like the one in [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en). In the next lesson, *Backpropagation by hand on a tiny graph*, you get every slope in one pass backwards through the calculation, for a few times the cost of a forward pass.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_flops, result_growth, choice_bigger):
    L.wrap_up(result_flops, result_growth, choice_bigger)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
