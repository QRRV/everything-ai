# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-model-is-a-function. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-model-is-a-function", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Examples, features and labels](/course/open/ml-vocabulary?lang=en) and [Functions](/course/open/py-functions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_feature():
    quick_feature = L.choice(
        "quick-feature",
        question="A model predicts the price of a house from its area. What is the area, in this task?",
        options={
            "label": "The label",
            "feature": "A feature",
            "example": "An example",
        },
        correct="feature",
        why={
            "label": "The label is what the model should predict. Is that the area, or the price? See the table of new words in [Examples, features and labels](/course/open/ml-vocabulary?lang=en).",
            "example": "An example is one whole row: one house. What is one value you know about it beforehand? See the table of new words in [Examples, features and labels](/course/open/ml-vocabulary?lang=en).",
            "feature": "The area is known beforehand, for every house: it is a feature. The price is the label.",
        },
    )
    quick_feature
    return (quick_feature,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="```python\ndef double(x):\n    return 2 * x\n```\n\nWhat does `double(4) + 1` give?",
        options={
            "ten": "`10`",
            "eight": "`8`",
            "nine": "`9`",
        },
        correct="nine",
        why={
            "ten": "That would be `double(4 + 1)`. What happens first here: the call, or the `+ 1`? See the recipe in [Functions](/course/open/py-functions?lang=en).",
            "eight": "That is `double(4)` on its own. What does `+ 1` do with what the function gives back? See the table of new words in [Functions](/course/open/py-functions?lang=en).",
            "nine": "`double(4)` gives back 8, and then `+ 1` makes 9.",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Examples, features and labels](/course/open/ml-vocabulary?lang=en), two houses had no price yet, one of them 88 square meters. How does a model get from 88 square meters to a price? A trained model can look like a black box, but inside it is a function: the features of an example go in, and a prediction comes out. What the model learned from its examples sits in a few numbers, its parameters. Change them, and the model predicts something else. This lesson looks at a very simple model: a straight line, with two parameters.

    **An everyday example.** A taxi fare: a fixed price to start, and a price for every kilometer. With those two numbers, you can work out the fare of any ride.
    """)
    return


@app.cell(hide_code=True)
def predict_house():
    guess_house = L.guess(
        "house-price",
        question="A model predicts the price of a house, in thousands of euros, from its area in square meters: price = 3 × area + 50. What do you think it predicts for a house of 100 square meters?",
        options={
            "no-bias": "300",
            "full": "350",
            "sum": "153",
        },
        reveal="**350.** 3 times 100 is 300, and 50 on top makes 350 thousand euros. The 3 says how much the price goes up for every square meter, and the 50 is added once, for every house.",
    )
    guess_house
    return (guess_house,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A model is a function. It gets the features of an example and gives back a prediction. With one feature, a very simple model is a straight line:
        """),
        L.formula(r"[[pred:\hat{y}]] = [[param:w]] \cdot \mathbf{x} + [[param:b]]"),
        mo.md(r"""
        The dot means times. In words: the <span class="c-pred">prediction</span> is the weight times the feature, plus the bias term. The two numbers <span class="c-param">$w$</span> and <span class="c-param">$b$</span> are the parameters of the model: they decide which prediction it makes, and they are what training will choose. In this lesson, you choose them yourself.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **model** | | a function that turns the features of an example into a prediction | `predict(x, w, b)` |
        | **parameter** | | a number inside a model that decides its predictions; training chooses it | `w`, `b` |
        | **weight**, $w$ | w | how much the prediction changes when the feature goes up by 1 | `w` |
        | **bias term**, $b$ | b | the prediction when the feature is 0: it is added once | `b` |
        | $\mathbf{x}$ | x | the feature of an example | `x` |
        | **prediction**, $\hat{y}$ | y hat | what the model gives for an example | `y_pred` |
        | $y$ | y | the label of an example: the right answer | `y` |

        On a chart, $w$ is how steep the line is, and $b$ is its height at $\mathbf{x} = 0$, where it crosses the vertical axis; in the chart below, that lies off to the left. With more features, each feature gets a weight of its own, and the model adds up each feature times its weight, plus one bias term.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_line():
    mo.vstack([
        mo.Html(C.picture({"title": "A model for the price of a house", "x": "area (m²)", "y": "price (thousand euros)",
                           "line": "3 × area + 50",
                           "alt": "Eight sold houses as points, area against price, and a straight line through them: the prediction 3 times the area plus 50. Most points lie close to the line."})),
        mo.md(r"""
        The eight houses that were sold, from [Examples, features and labels](/course/open/ml-vocabulary?lang=en), as points: area against price. The <span class="c-pred">line</span> is the model with $w = 3$ and $b = 50$: for every area, the line gives the prediction. Most houses lie close to it, a few further away. Other values of $w$ and $b$ give another line, and other predictions.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Two words mean something else here than you may expect. A *parameter* is a number inside a model, such as $w$ or $b$; the values you hand to a Python function are its arguments, as in [Functions](/course/open/py-functions?lang=en), even though Python's own documentation calls the names in a `def` line parameters. And the *bias term* is just the number $b$; it has nothing to do with being unfair, which is another meaning of bias."),
        mo.md(r"""
        ### Recipe

        To make a model and use it:

        1. Write the model as a Python function: the features and the parameters go in, the prediction comes out.
        2. Choose values for the parameters.
        3. Call the function with the features of an example: that gives its prediction.
        4. Draw the line of the model over the examples, to see how close it comes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The model from the picture, for the houses. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_model():
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. the model: feature and parameters in, prediction out
    def _predict(x, w, b):
        return w * x + b

    # 2. the parameters
    _w = 3
    _b = 50
    # 3. the prediction for a house of 100 square meters
    print(_predict(100, _w, _b))
    # 3. and for the eight sold houses at once: x is an array
    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])
    print(_predict(_area, _w, _b))
    # 4. the houses as points, and the line of the model
    plt.scatter(_area, _price)
    plt.plot([40, 140], [_predict(40, _w, _b), _predict(140, _w, _b)])
    plt.xlabel("area (m²)")
    plt.ylabel("price (thousand euros)")
    plt.title("A model for the price of a house")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A house of 100 square meters gets the prediction 350, as in the guess. With an array of areas, the same function gives a prediction for each house at once: `w * x` multiplies every area, as in vectorization. Put the predictions next to the real prices: the house of 62 square meters gets 236 and sold for 245, the one of 130 gets 440 and sold for 465. A line needs only two points, so the chart draws it from 40 to 140 square meters.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weight():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What a weight says
    """)
    return


@app.cell(hide_code=True)
def choice_weight_cell():
    choice_weight = L.choice(
        "weight-meaning",
        question="A model predicts the price of a used car, in euros, from its age in years: price = −1500 × age + 20000. What does the weight, −1500, say?",
        options={
            "new": "A new car costs 1500 euros",
            "per-year": "Every year older makes the predicted price 1500 euros lower",
            "higher": "Every year older makes the predicted price 1500 euros higher",
        },
        correct="per-year",
        why={
            "higher": "Look at the sign of the weight. What does −1500 × age do when the age goes up by 1?",
            "new": "A new car has age 0. What is the prediction then: which parameter is left?",
            "per-year": "The weight is how much the prediction changes when the feature goes up by 1: one year older, 1500 euros less. The bias term, 20000, is the prediction for a car of age 0.",
        },
    )
    choice_weight
    return (choice_weight,)


@app.cell(hide_code=True)
def text_exercise_taxi():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The parameters of a taxi

        A taxi charges 4 euros to start, and 2.5 euros for every kilometer. The box has the model `fare(km, w, b)`. Set its two parameters, `w` and `b`, so that it gives the fare of every ride. Replace the two `...`, and leave the other lines as they are. The last line works out a ride of 12 kilometers. The check also tries a ride of another length.
        """),
        L.task("Set `w` and `b` so that `fare(km, w, b)` gives the fare of a ride."),
    ])
    return


@app.cell(hide_code=True)
def exercise_taxi():
    box_taxi = L.code_box("taxi", lines=9)
    box_taxi
    return (box_taxi,)


@app.cell(hide_code=True)
def check_taxi(box_taxi):
    result_taxi = L.check("taxi", box_taxi.value)
    result_taxi
    return (result_taxi,)


@app.cell(hide_code=True)
def hints_taxi():
    L.hints(
        "taxi",
        [
            "Step 2 of the recipe: choose the parameters. Which price is paid for every kilometer, and which one only once?",
            "The weight `w` is multiplied by the kilometers; the bias term `b` is added once.",
            "`w = ___` (what one kilometer costs) and `b = ___` (what you pay once).",
        ],
        why="`w = 2.5` is the price of every kilometer, and `b = 4` the start price. 12 kilometers then cost 2.5 × 12 + 4 = 34 euros.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A model with two features

        A better model for houses uses two features: the area and the number of rooms. With the parameters of the check, it is closer to the eight prices than the line of the picture. Each feature gets its own weight, `w_area` and `w_rooms`, and there is one bias term, `b`. Finish `predict(area, rooms, w_area, w_rooms, b)`: it gives back the prediction. Replace the `...`. The check calls `predict(100, 4, 2, 25, 45)`: that should give `345`. It also tries other houses and other parameters.
        """),
        L.task("Make `predict` give back the prediction of a model with two features."),
    ])
    return


@app.cell(hide_code=True)
def exercise_two():
    box_two = L.code_box("two-features", lines=5)
    box_two
    return (box_two,)


@app.cell(hide_code=True)
def check_two(box_two):
    result_two = L.check("two-features", box_two.value)
    result_two
    return (result_two,)


@app.cell(hide_code=True)
def hints_two():
    L.hints(
        "two-features",
        [
            "Like the model with one feature, but now every feature brings its own part.",
            "Three parts are added up: one for the area, one for the rooms, and the bias term. Which weight goes with which feature?",
            "`return ___ * area + ___ * rooms + ___`.",
        ],
        why="`w_area * area` is the part of the area, `w_rooms * rooms` the part of the rooms, and `b` is added once: 2 × 100 + 25 × 4 + 45 = 345.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Examples, features and labels](/course/open/ml-vocabulary?lang=en) you met features and labels; a model is the function that goes from the one to the other. In [Functions](/course/open/py-functions?lang=en) you wrote functions that get arguments and give back a value; a model is such a function, with its parameters as extra arguments. In [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) you drew a line; the line of a model shows all its predictions at once. In the next lesson, *How wrong is one prediction?*, you measure how far a prediction is from the label.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weight, result_taxi, result_two):
    L.wrap_up(choice_weight, result_taxi, result_two)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
