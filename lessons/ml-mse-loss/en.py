# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-mse-loss. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-mse-loss", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How wrong is one prediction?](/course/open/ml-residuals?lang=en) and [Calculating without loops](/course/open/np-vectorization?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="A model predicted that a ride would take 52 minutes. It took 47. What is the residual?",
        options={
            "flipped": "5",
            "right": "−5",
            "label": "47",
        },
        correct="right",
        why={
            "flipped": "That is the prediction minus the label. In which order does a residual subtract? See the formula in [How wrong is one prediction?](/course/open/ml-residuals?lang=en).",
            "label": "That is the label itself. A residual says how far the prediction is off. See the formula in [How wrong is one prediction?](/course/open/ml-residuals?lang=en).",
            "right": "The label minus the prediction: 47 − 52 = −5. Negative: the model predicted 5 minutes too long.",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def quick_check_square():
    quick_square = L.choice(
        "quick-square",
        question="`a = np.array([1, -2, 3])`. What does `a ** 2` give?",
        options={
            "right": "`array([1, 4, 9])`",
            "sign": "`array([1, -4, 9])`",
            "times": "`array([2, -4, 6])`",
        },
        correct="right",
        why={
            "sign": "What is −2 times −2? See the table of new words in [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "times": "That is `a * 2`. What does `**` do with a number? See the table of new words in [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "right": "`** 2` squares every number of the array: 1, 4 and 9. Minus times minus is plus, so a square is never negative.",
        },
    )
    quick_square
    return (quick_square,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your model predicts the price of five houses, and for each house it is a bit off: five <span class="c-residual">residuals</span>. Now you have a second model, with five residuals of its own. Which model is better? With ten separate numbers, you cannot say. You want **one number** per model.

    **An everyday example.** Two weather forecasters predict the temperature for three days. This is how far off they are, in degrees:

    | | Monday | Tuesday | Wednesday |
    |---|---|---|---|
    | Forecaster A | +3 | −3 | 0 |
    | Forecaster B | +1 | +1 | +1 |
    """)
    return


@app.cell(hide_code=True)
def predict_forecasters():
    guess_forecaster = L.guess(
        "forecasters",
        question="Which forecaster do you think predicts better?",
        options={
            "a": "A",
            "b": "B",
            "same": "Equally good",
        },
        reveal="**Take the plain mean of the residuals, and A wins: 0 against 1.** But A was 3 degrees off on two of the three days, and B never more than 1. In a plain mean, plus and minus cancel each other out. This lesson makes one number in which they do not.",
    )
    guess_forecaster
    return (guess_forecaster,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The loss is one number that says how far a model is off over all examples together: lower is better. A common loss is the mean squared error: square each residual, and take the mean.
        """),
        L.formula(r"[[loss:\mathcal{L}]] = \frac{1}{n} \sum_{i=1}^{n} [[residual:(y_i - \hat{y}_i)]]^2"),
        mo.md(r"""
        In words: for each example $i$, the <span class="c-residual">residual</span> $y_i - \hat{y}_i$, squared; add those up for all $n$ examples, and divide by $n$. That is the <span class="c-loss">loss</span> $\mathcal{L}$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **loss**, $\mathcal{L}$ | L, the loss | one number that says how far a model is off over all examples together; lower is better | `loss` |
        | **mean squared error** (MSE) | | a loss: the mean of the squared residuals | `mse` |
        | $n$ | n | the number of examples | `n`, `len(y)` |
        | $y_i$, $\hat{y}_i$ | y i, y hat i | the label and the prediction of example $i$ | `y[i]`, `y_pred[i]` |
        | $\sum_{i=1}^{n}$ | sum over i from 1 to n | add up what comes after it, for every $i$ from 1 to $n$ | `.sum()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_squares():
    mo.vstack([
        mo.Html(C.picture({"title": "The squared residuals", "x": "area (m²)", "y": "price (thousand euros)",
                           "alt": "Five houses as points and the line of a model. From the line to each point runs a vertical stroke, the residual, and on each stroke stands a square with that stroke as its side. The square of the house of 95 square meters, with a residual of 41, is much bigger than all the others."})),
        mo.md(r"""
        Five houses and the <span class="c-pred">line</span> of a model, with each <span class="c-residual">residual</span> as a stroke. On each stroke stands a <span class="c-loss">square</span> with that stroke as its side: its area is the residual squared. The mean squared error is the mean area of the squares. The house of 95 square meters has a residual of 41, and a square of 1681: more than the other four squares together, 901.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_why_square():
    mo.vstack([
        mo.md(r"""
        Squaring does two things. A square is never negative, so plus and minus no longer cancel each other out. And a big miss counts extra: a residual of 4 gives 16, four times as much as a residual of 2, which gives 4.

        Back to the forecasters:

        - A: $(3^2 + (-3)^2 + 0^2) / 3 = 18 / 3 = 6$
        - B: $(1^2 + 1^2 + 1^2) / 3 = 1$

        With the mean squared error, B wins.
        """),
        L.note("**Why not just drop the minus sign?** You can: the mean of the residuals without their sign is a loss too. The square is used more often, because it is easier to calculate with later on. You will see why once a model learns by itself."),
        L.watch_out("The *error* in mean squared error means the residual: how far a prediction is off. It has nothing to do with an error in your code."),
        L.watch_out("Some books call the squared residual of one example the loss, and the mean over all examples the *cost*. In this course, both are the loss."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To work out the mean squared error of a model:

    1. Work out the residual of each example: $y_i - \hat{y}_i$.
    2. Square each residual.
    3. Take the mean: add up the squares and divide by the number of examples, $n$.

    The formula counts the examples from 1, Python from 0: $y_1$ is `y[0]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three houses, with prices in thousands of euros. One residual is positive, one negative and one 0. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_houses():
    _y = [300, 250, 400]       # the real prices
    _y_pred = [280, 260, 400]  # what the model predicted
    # 1. the residual of each house
    _residuals = []
    for _i in range(len(_y)):
        _residuals.append(_y[_i] - _y_pred[_i])
    print(_residuals)
    # 2. square each residual
    _squares = []
    for _r in _residuals:
        _squares.append(_r ** 2)
    print(_squares)
    # 3. the mean: add up, and divide by the number of houses
    _total = 0
    for _s in _squares:
        _total = _total + _s
    _mse = _total / len(_squares)
    print(_mse)
    return


@app.cell(hide_code=True)
def text_numpy():
    mo.md(r"""
    The residuals are 20, −10 and 0; squared, 400, 100 and 0; together 500, and divided by 3 houses, 166.67. That number is large because of the squares. On its own it says little; next to the loss of another model on the same houses, it says which of the two is better.

    **The same in NumPy.** With arrays, each step of the recipe works on all houses at once, without a loop:

    | Step | Plain Python | NumPy |
    |---|---|---|
    | 1 | a loop with `_residuals.append(_y[_i] - _y_pred[_i])` | `_y - _y_pred` |
    | 2 | a loop with `_squares.append(_r ** 2)` | `_residuals ** 2` |
    | 3 | a loop that adds up, then `/ len(_squares)` | `_squares.mean()` |
    """)
    return


@app.cell
def example_houses_numpy():
    import numpy as np

    _y = np.array([300, 250, 400])
    _y_pred = np.array([280, 260, 400])
    _residuals = _y - _y_pred    # 1. the residual of each house
    _squares = _residuals ** 2   # 2. square
    _mse = _squares.mean()       # 3. the mean
    print(_mse)
    return


@app.cell(hide_code=True)
def text_exercise_temps():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The loss of a forecaster

        A forecaster predicted the temperature for four days. The box has the real temperatures as the list `temps_y`, and the predictions as `temps_y_pred`. Work out the mean squared error of the forecaster, and put it in `mse_temps`. Replace the `...`, with plain Python or with NumPy. The last line is `mse_temps`, so the check sees it. The check also tries your code on two other days.
        """),
        L.task("Put the mean squared error of the four days in `mse_temps`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_temps():
    box_temps = L.code_box("mse-temps", lines=14)
    box_temps
    return (box_temps,)


@app.cell(hide_code=True)
def check_temps(box_temps):
    result_temps = L.check("mse-temps", box_temps.value)
    result_temps
    return (result_temps,)


@app.cell(hide_code=True)
def hints_temps():
    L.hints(
        "mse-temps",
        [
            "Step 1 of the recipe: first the residual of each day, with a loop, as with the three houses.",
            "There are four days; `for i in range(len(temps_y))` goes through all of them. Then the squares, then their mean.",
            "`residuals.append(temps_y[i] - ___)`.",
        ],
        why="The residuals are −3, 0, 1 and −4. Their squares, 9, 0, 1 and 16, add up to 26, and 26 divided by 4 days is 6.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The mean squared error as a function

        Finish `mse(y, y_pred)`. It gets the labels and the predictions of some examples, as two lists of the same length, and gives back one number: the mean squared error. Its first two lines make arrays of the lists. Replace the `...`. The check calls `mse([3, 5, 2, 6], [4, 5, 0, 6])`: that should give `1.25`. It also tries other examples.
        """),
        L.task("Make `mse` give back the mean squared error of `y` and `y_pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mse():
    box_mse = L.code_box("mse", lines=9)
    box_mse
    return (box_mse,)


@app.cell(hide_code=True)
def check_mse(box_mse):
    result_mse = L.check("mse", box_mse.value)
    result_mse
    return (result_mse,)


@app.cell(hide_code=True)
def hints_mse():
    L.hints(
        "mse",
        [
            "The recipe has three steps, and in NumPy each step is one operation on the whole array.",
            "Step 1 subtracts the two arrays, and step 2 squares with `** 2`. Which method from [Sums and means along an axis](/course/open/np-axis?lang=en) does step 3?",
            "`return ((y - y_pred) ** 2).___()`.",
        ],
        why="`y - y_pred` gives the residuals, `** 2` squares each of them, and `.mean()` takes their mean. For the example: residuals −1, 0, 2 and 0, squares 1, 0, 4 and 0, mean 5 / 4 = 1.25. Whether you take `y - y_pred` or `y_pred - y` makes no difference once you square.",
    )
    return


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Exercise 3. Why square?
    """)
    return


@app.cell(hide_code=True)
def explain_square_cell():
    explain_square = L.explain(
        "why-square",
        question="Why does the mean squared error square the residuals, instead of just taking their mean? Write one or two sentences.",
        model_answer="In the plain mean, positive and negative residuals cancel each other out, so a model that is far off in both directions can still get 0. A square is never negative, so every miss counts, and a big miss counts extra.",
        points=[
            "in a plain mean, plus and minus cancel each other out",
            "a square is never negative, so every miss counts",
            "a big miss counts extra",
        ],
    )
    explain_square
    return (explain_square,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [How wrong is one prediction?](/course/open/ml-residuals?lang=en) you worked out one residual per example; the loss puts all of them together in one number, so you can compare whole models. With arrays, as in [Calculating without loops](/course/open/np-vectorization?lang=en), each step of the recipe is one operation, and `.mean()`, from [Sums and means along an axis](/course/open/np-axis?lang=en), takes the mean. In the next lesson, *Fitting a line by hand*, you move the parameters yourself and watch the loss: the best line is the one with the lowest loss.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(explain_square, result_mse, result_temps):
    L.wrap_up(result_temps, result_mse, explain_square)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
