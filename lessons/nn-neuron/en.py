# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-neuron. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-neuron", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) and [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logit():
    quick_logit = L.choice(
        "quick-logit",
        question="Logistic regression has the weights $(2, -1)$ and the bias term $0.5$. An example has the features $(1, 3)$. Is the probability it predicts above or below 0.5?",
        options={
            "above": "Above 0.5: the weighted sum plus the bias term is 5.5",
            "right": "Below 0.5: the weighted sum plus the bias term is −0.5",
            "equal": "Exactly 0.5: the example lies on the decision boundary",
        },
        correct="right",
        why={
            "above": "$2 \\cdot 1 + 1 \\cdot 3 + 0.5$ would be 5.5, but the second weight is −1. See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "equal": "On the decision boundary the weighted sum plus the bias term is exactly 0. What is it here? See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "right": "$2 \\cdot 1 - 1 \\cdot 3 + 0.5 = -0.5$: below 0, so the sigmoid gives less than 0.5.",
        },
    )
    quick_logit
    return (quick_logit,)


@app.cell(hide_code=True)
def quick_check_line():
    quick_line = L.choice(
        "quick-line",
        question="What does linear regression predict for an example with the features `x`?",
        options={
            "sigmoid": "The sigmoid of `np.dot(w, x) + b`",
            "class": "1 or 0, whichever is closer",
            "right": "`np.dot(w, x) + b` itself",
        },
        correct="right",
        why={
            "sigmoid": "That is logistic regression, a probability. What does linear regression predict, a price for example? See [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "class": "Linear regression predicts a number, such as a price. See [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "right": "The weighted sum plus the bias term, with nothing after it: any number.",
        },
    )
    quick_line
    return (quick_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every neural network in this course, up to the language model at the end, is built from one small part: the neuron. You have met it already, without the name. The name comes from the nerve cells in a brain, which also collect signals and pass one on when they get enough; the likeness stops there. A neuron in a network is a few lines of arithmetic.

    **An everyday example.** A smoke alarm does not go off at the first wisp of smoke. It weighs what its sensors measure, smoke and heat, and only beeps once the total is above a set level.
    """)
    return


@app.cell(hide_code=True)
def predict_fires():
    guess_fires = L.guess(
        "fires",
        question="A neuron has the weights 1 and 1 and the bias term −1.5. It gives 1 when its weighted sum plus the bias term is above 0, and 0 otherwise. For which of the inputs (0, 0), (0, 1), (1, 0) and (1, 1) do you think it gives 1?",
        options={
            "one": "Only for (1, 1)",
            "three": "For (0, 1), (1, 0) and (1, 1)",
            "none": "For none of them",
            "all": "For all four",
        },
        reveal="**Only for (1, 1).** There the sum is 1 + 1 − 1.5 = 0.5, above 0. With one 1 it is −0.5, and with none −1.5. The neuron gives 1 only when both inputs are 1: it computes AND.",
    )
    guess_fires
    return (guess_fires,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **neuron** multiplies each input by its weight, adds them up with a bias term, and puts the result through an **activation function** $f$:
        """),
        L.formula(r"z = [[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]] \qquad [[pred:\hat{y}]] = f(z)"),
        mo.Html(C.diagram({"sum": "weighted sum", "activation": "activation function", "output": "output",
                           "alt": "A neuron with two inputs, x₁ and x₂. An arrow from each, with its weight w₁ or w₂, goes to a circle Σ that adds them up with the bias term b. Its result z goes into a box f, the activation function, and out comes the output ŷ."})),
        mo.md(r"""
        The activation function decides what the neuron does with $z$. Three of them, each for a model you know:

        | Activation function | $f(z)$ | The neuron is then |
        |---|---|---|
        | none | $z$ | linear regression: any number |
        | sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | logistic regression: a probability |
        | step | 1 when $z > 0$, else 0 | a yes or no: the neuron fires or not |

        So logistic regression is one neuron, and so is linear regression. A neural network is more than one neuron: it connects many, the output of one becoming the input of the next; the next lessons build that up. Learning, for a network, is what it was for linear regression: adjusting the weights and bias terms.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **neuron** | | the smallest part of a neural network: a weighted sum of its inputs plus a bias term, put through an activation function | `f(np.dot(w, x) + b)` |
        | **activation function** | | the function a neuron puts its weighted sum through, such as the sigmoid or a step | `1 / (1 + np.exp(-z))` |
        | $f$ | f | the activation function of a neuron | |
        | **fires** | | a neuron with a step gives 1 | `z > 0` |
        """),
        L.watch_out("A neuron in a network is only loosely like a nerve cell in a brain. It is arithmetic, and the course never needs more biology than the name."),
        mo.md(r"""
        Move the slider: it sets the bias term $b$ of a neuron with the weights 1 and 1 and a step. The four corners of the square are its inputs: a filled dot where the neuron fires, an open one where it does not. The <span class="c-pred">line</span> is where $z = 0$: the neuron fires on its upper right side. The other chart counts, for every $b$, the corners that fire; drag across it to set $b$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        b = round(b, 1)
        fires = C.fires(b)
        which = ", ".join(str(c) for c, f in zip(C.CORNERS, fires) if f) or "none of them"
        return (C.square(b, {"title": "Which corners fire?", "count_title": "Corners that fire, for every b", "count": "corners",
                              "alt": f"The four corners of the square with b = {L.number(b, 1)}: the neuron fires for {which}. The line x₁ + x₂ + b = 0 runs from top left to bottom right, with the firing side to the top right. A second chart counts the corners that fire for every b: none up to and including −2, one up to and including −1, three up to and including 0, and four above 0."}),
                f"""With <i>b</i> = {L.number(b, 1)} the neuron fires for <span class="c-pred ea-num">{sum(fires)}</span> of the 4 corners: {which}.""")

    explore_b = L.explore("bias", "b", _frame, start=-1.5, low=-2.5, high=0.5, step=0.1, role="param", task="Find the values of $b$ for which the neuron fires for every corner except (0, 0).")
    explore_b
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The bias term shifts the line: up to and including −2 nothing fires, then up to and including −1 only (1, 1), then up to and including 0 three corners, every one except (0, 0), and above 0 all four. The count never stops at 2: (0, 1) and (1, 0) have the same $z$, so they start to fire together. The weights decide the direction of the line; the bias term decides where it lies.

    ### Recipe

    To work out what a neuron gives for the inputs `x`:

    1. Multiply each input by its weight and add up: `np.dot(w, x)`.
    2. Add the bias term: `z = np.dot(w, x) + b`.
    3. Put `z` through the activation function: a step, `1` when `z > 0` and else `0`; the sigmoid; or nothing at all.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The neuron of the guess, with the weights 1 and 1 and the bias term −1.5, on all four inputs, with a step and with the sigmoid as its activation function. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_neuron():
    import numpy as np

    _w = np.array([1, 1])
    _b = -1.5
    for _x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        # 1. and 2. the weighted sum plus the bias term
        _z = np.dot(_w, _x) + _b
        # 3. a step, and the sigmoid
        if _z > 0:
            _step = 1
        else:
            _step = 0
        _sigmoid = 1 / (1 + np.exp(-_z))
        print(_x, _z, _step, _sigmoid)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $z$ is −1.5 for (0, 0), −0.5 for (0, 1) and (1, 0), and 0.5 for (1, 1). With a step, only (1, 1) gives 1: AND. With the sigmoid, the same neuron gives probabilities instead: about 0.18 and 0.38 below the line, and 0.62 above it. That is logistic regression, with these weights and this bias term.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_activation():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which activation function?
    """)
    return


@app.cell(hide_code=True)
def choice_activation_cell():
    choice_activation = L.choice(
        "which-activation",
        question="A neuron should predict the price of a house from its area and its distance to the station. Which activation function fits?",
        options={
            "sigmoid": "The sigmoid, so that the price comes out between 0 and 1",
            "step": "A step, so that the neuron fires for the expensive houses and stays quiet for the rest",
            "right": "None: the weighted sum plus the bias term is the price, as in linear regression",
        },
        correct="right",
        why={
            "sigmoid": "A sigmoid squeezes everything between 0 and 1. Can a price be 300,000 euros then? See the table in *The idea*.",
            "step": "A step gives only 1 or 0: expensive or not, but no price. See the table in *The idea*.",
            "right": "A price can be any number, so nothing should squeeze or cut $z$: the neuron is then linear regression.",
        },
    )
    choice_activation
    return (choice_activation,)


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A neuron with a step

        Finish `step_neuron(x, w, b)`: it gets the inputs `x`, the weights `w` and the bias term `b`, and gives back 1 when the weighted sum plus the bias term is above 0, and 0 otherwise. Replace the two `...` with steps 2 and 3 of the recipe. The check calls `step_neuron([1, 1], [1, 1], -1.5)`, the neuron of the worked example: that should give `1`. It also tries other inputs, weights and bias terms, also with three inputs, and one where `z` is exactly 0.
        """),
        L.task("Make `step_neuron` give back 1 when `np.dot(w, x) + b` is above 0, and 0 otherwise."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("step-neuron", lines=8)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("step-neuron", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "step-neuron",
        [
            "Steps 2 and 3 of the recipe, as in the worked example: first `z`, then the `if`.",
            "`z` is a dot product plus the bias term. Then an `if`: which of the two numbers comes back when `z` is above 0?",
            "`z = np.dot(___, ___) + b`, then `if z > 0:` with `return ___`, and `else:` with `return ___`.",
        ],
        why="`np.dot(w, x) + b` is $z$: for `[1, 1]` with the weights 1 and 1 and −1.5, it is 0.5. That is above 0, so the neuron gives 1. Exactly at 0 it gives 0: the step fires only above 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_corner():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Only one corner

        Choose the weights `w` and the bias term `b` of a neuron with a step, so that it fires only for the input (1, 0): the first input on, the second off. Replace the two `...`; the loop below works out the output for all four corners, in the order (0, 0), (0, 1), (1, 0), (1, 1), so that should give `[0, 0, 1, 0]`. Leave the loop as it is. Want to try a few? Do that in the scratchpad.
        """),
        L.task("Choose `w` and `b` so that the neuron fires only for (1, 0)."),
    ])
    return


@app.cell(hide_code=True)
def exercise_corner():
    box_corner = L.code_box("one-corner", lines=13)
    box_corner
    return (box_corner,)


@app.cell(hide_code=True)
def check_corner(box_corner):
    result_corner = L.check("one-corner", box_corner.value)
    result_corner
    return (result_corner,)


@app.cell(hide_code=True)
def hints_corner():
    L.hints(
        "one-corner",
        [
            "With the weights 1 and 1, the slider at its start, b = −1.5, showed AND. Here the second input should count against the neuron. What sign does its weight need?",
            "For (0, 0), `z` is just `b`: is that above or below 0? For (1, 0), `z` is the first weight plus `b`. And for (1, 1)?",
            "`w = np.array([1, ___])` and `b = ___`: a `b` between −1 and 0 works.",
        ],
        why="With `w = np.array([1, -1])` and `b = -0.5`, `z` is −0.5 for (0, 0), −1.5 for (0, 1), 0.5 for (1, 0) and −0.5 for (1, 1): only (1, 0) is above 0. Many other choices work too, as long as (1, 0) is the only corner on the side of the line where the neuron fires.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) you met a neuron with a sigmoid without the name, and in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) one with no activation function at all. A neuron draws one straight line through its inputs, and fires on one side of it. In the next lesson, *What one neuron cannot learn*, you meet a task that no straight line can solve.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_activation, result_corner, result_step):
    L.wrap_up(choice_activation, result_step, result_corner)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
