# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-residuals. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-residuals", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_predict():
    quick_predict = L.choice(
        "quick-predict",
        question="A model is $\\hat{y} = 2 \\cdot \\mathbf{x} + 10$. What does it predict for $\\mathbf{x} = 4$?",
        options={
            "no-bias": "8",
            "right": "18",
            "added": "16",
        },
        correct="right",
        why={
            "no-bias": "That is only 2 times 4. What is added once, for every prediction? See the table of new words in [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "added": "That adds 2, 4 and 10. What does the dot between 2 and $\\mathbf{x}$ mean? See *The idea* in [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "right": "2 times 4 is 8, and the bias term 10 on top makes 18.",
        },
    )
    quick_predict
    return (quick_predict,)


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="In the model $\\hat{y} = w \\cdot \\mathbf{x} + b$, which are the parameters?",
        options={
            "wb": "$w$ and $b$",
            "xy": "$\\mathbf{x}$ and $\\hat{y}$",
            "w": "Only $w$",
        },
        correct="wb",
        why={
            "xy": "$\\mathbf{x}$ is the feature, and $\\hat{y}$ the prediction. Which numbers decide which prediction the model makes? See the table of new words in [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "w": "$w$ is one of them. Is there another number that changes every prediction? See the table of new words in [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "wb": "The weight $w$ and the bias term $b$ decide which prediction the model makes: they are its parameters.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model makes a prediction, and for an example with a label you can see how good it is. Before you can say whether one model is better than another, or improve one, you need a number for how far a prediction is off. That number, for one example, is the first step towards everything a model learns.

    **An everyday example.** A weather forecast said 18 degrees, and it was 21. The forecast was 3 degrees too low. Another day it said 15 and it was 13: 2 degrees too high.
    """)
    return


@app.cell(hide_code=True)
def predict_sign():
    guess_sign = L.guess(
        "sign",
        question="A model predicted 280 thousand euros for a house that sold for 300. Another time it predicted 340 for a house that sold for 320. Do you think a number for how far off a prediction is should be the same for both?",
        options={
            "same": "Yes: both are 20 off",
            "sign": "No: the first was too low, the second too high",
        },
        reveal="**Both are 20 off, but in different directions.** The residual keeps that direction in its sign: 300 − 280 = 20, too low, and 320 − 340 = −20, too high. How far off, and which way, in one number.",
    )
    guess_sign
    return (guess_sign,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        For one example with a label, the residual is how far the prediction is off: the label minus the prediction.
        """),
        L.formula(r"[[residual:\text{residual}]] = y - [[pred:\hat{y}]]"),
        mo.md(r"""
        A positive <span class="c-residual">residual</span> means the model predicted too low; a negative one, too high; 0, exactly right. It is in the unit of the label: here, thousand euros.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **residual** | | for one example, the label minus the prediction: how far that one prediction is off, and which way | `y - y_pred` |
        | `.argmin()` | | like `.argmax()`, but the place of the smallest number | `residuals.argmin()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_residuals():
    mo.vstack([
        mo.Html(C.picture({"title": "The residuals of the houses", "x": "area (m²)", "y": "price (thousand euros)", "biggest": "+25",
                           "alt": "The eight sold houses as points, the line of the model 3 times the area plus 50, and from the line to each point a vertical stroke: the residual. Seven points lie above the line; the house of 95 square meters lies 15 below it. The longest stroke, +25, belongs to the house of 130 square meters."})),
        mo.md(r"""
        The houses and the model from [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en), with each <span class="c-residual">residual</span> as a stroke from the <span class="c-pred">line</span> to the point. A house above the line sold for more than predicted: a positive residual. Seven houses lie above the line; the one of 95 square meters lies below it. The longest stroke, 25, belongs to the house of 130 square meters.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Many books call the residual the *error*. In this course, how far a prediction is off is always the residual; an error is something that goes wrong, for example in your code."),
        mo.md(r"""
        ### Recipe

        To work out a residual:

        1. Make the prediction $\hat{y}$ of the example with the model.
        2. Subtract it from the label: $y - \hat{y}$, in that order.
        3. For many examples at once, do the same with arrays: one residual per example.
        4. Read the sign: positive is too low, negative too high.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The residuals of the eight houses, with the model of the picture. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_residuals():
    import numpy as np

    _w = 3
    _b = 50
    # 1. one house: 130 square meters, sold for 465
    _y_pred = _w * 130 + _b
    print(_y_pred)
    # 2. its residual
    print(465 - _y_pred)
    # 3. all eight houses at once
    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])
    _residuals = _price - (_w * _area + _b)
    print(_residuals)
    # 4. the house with the most negative residual: predicted too high
    print(_residuals.argmin())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The house of 130 square meters gets the prediction 440 and sold for 465: a residual of 25, the model was 25 thousand euros too low. For all houses at once, `_price - (_w * _area + _b)` subtracts, place by place, as in vectorization; the brackets make sure the whole prediction is subtracted. Seven residuals are positive. `.argmin()` gives place 4, the house of 95 square meters, with a residual of −15: the only one the model predicted too high.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One residual
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-residual",
        question="A model predicts that a ride will take 25 minutes. It took 31. What is the residual?",
        options={
            "neg": "−6",
            "pred": "25",
            "pos": "6",
        },
        correct="pos",
        why={
            "neg": "That is the prediction minus the label. In which order does a residual subtract? See the formula in *The idea*.",
            "pred": "That is the prediction itself. A residual is how far it is off. See the formula in *The idea*.",
            "pos": "The label minus the prediction: 31 − 25 = 6. Positive: the model predicted too low, 6 minutes too short.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_residuals():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. All residuals

        The box has five rides of the taxi from [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en): the distances as the array `km`, and the fares that were paid as `fare`. The model is the one of that lesson, with `w = 2.5` and `b = 4`; real fares differ a little, for example because of traffic. Make `residuals`: the residual of each ride, as an array. Replace the `...`. The last line is `residuals`, so the check sees it. The check also tries other parameters.
        """),
        L.task("Make `residuals`: the fare of each ride minus its prediction."),
    ])
    return


@app.cell(hide_code=True)
def exercise_residuals():
    box_residuals = L.code_box("residuals", lines=9)
    box_residuals
    return (box_residuals,)


@app.cell(hide_code=True)
def check_residuals(box_residuals):
    result_residuals = L.check("residuals", box_residuals.value)
    result_residuals
    return (result_residuals,)


@app.cell(hide_code=True)
def hints_residuals():
    L.hints(
        "residuals",
        [
            "Step 3 of the recipe: the same subtraction as for one example, but with whole arrays.",
            "First the predictions of the model, with `w` and `b`; then subtract. Which comes first in a residual, the label or the prediction?",
            "`residuals = fare - (___)`.",
        ],
        why="`w * km + b` gives the five predictions, and `fare - (...)` subtracts them from the fares, place by place. Without the brackets, Python would subtract only `w * km` and then add `b`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Predicted too low by the most

        Finish `too_low(area, price, w, b)`. It gets the areas and prices of some houses, and the parameters of a model. It gives back the place, from 0, of the house whose price the model predicts too low by the most. Replace the `...`. The check calls `too_low` with the eight houses and $w = 3$, $b = 50$: that should give `5`. It also tries other houses and parameters.
        """),
        L.task("Make `too_low` give back the place of the house the model predicts too low by the most."),
    ])
    return


@app.cell(hide_code=True)
def exercise_low():
    box_low = L.code_box("too-low", lines=9)
    box_low
    return (box_low,)


@app.cell(hide_code=True)
def check_low(box_low):
    result_low = L.check("too-low", box_low.value)
    result_low
    return (result_low,)


@app.cell(hide_code=True)
def hints_low():
    L.hints(
        "too-low",
        [
            "Two steps: first the residuals of all houses, then the place of the house the model predicts too low by the most.",
            "Predicted too low means a positive residual: which residual is it, the largest or the smallest? [Sums and means along an axis](/course/open/np-axis?lang=en) has the method that gives its place.",
            "`residuals = ___`, and then `return residuals.___()`.",
        ],
        why="The residuals are `price - (w * area + b)`. The house predicted too low by the most has the largest residual, and `.argmax()` gives its place.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en) a model made predictions; now you know for each example how far its prediction is off, and which way. With arrays, as in [Calculating without loops](/course/open/np-vectorization?lang=en), you get all residuals at once, and with `argmax`, from [Sums and means along an axis](/course/open/np-axis?lang=en), you find the largest. In the next lesson, *One number for how wrong: the loss*, you put all residuals together into one number, to compare whole models.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, result_low, result_residuals):
    L.wrap_up(choice_one, result_residuals, result_low)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
