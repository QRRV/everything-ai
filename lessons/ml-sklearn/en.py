# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-sklearn. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-sklearn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) and [Tables with pandas](/course/open/data-pandas?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="Your `fit(X, y, lr, steps)` from [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en): what does it give back?",
        options={
            "predictions": "The predictions for every example",
            "right": "The weights `w` and the bias term `b`",
            "losses": "The loss after every step",
        },
        correct="right",
        why={
            "predictions": "Predicting is a separate step, with `X @ w + b`. What does training find? See exercise 3 of [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "losses": "The loss goes down along the way, but what comes out of the function at the end? See exercise 3 of [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "right": "Training finds the parameters: `return w, b`. With those, `X @ w + b` predicts.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def quick_check_columns():
    quick_columns = L.choice(
        "quick-columns",
        question="Which gives a table with only the columns `area` and `garden` of `df`?",
        options={
            "right": "`df[[\"area\", \"garden\"]]`",
            "one-pair": "`df[\"area\", \"garden\"]`",
            "plus": "`df[\"area\"] + df[\"garden\"]`",
        },
        correct="right",
        why={
            "one-pair": "With one pair of brackets, pandas looks for one column with both names in it. See the watch-out in [Tables with pandas](/course/open/data-pandas?lang=en).",
            "plus": "That adds the two columns up, number by number: one column. See the table in [Tables with pandas](/course/open/data-pandas?lang=en).",
            "right": "The names in a list, inside the brackets: two pairs of brackets, and you get a new table with those two columns.",
        },
    )
    quick_columns
    return (quick_columns,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        You wrote linear regression yourself: the slopes, a loop, a learning rate, a number of steps. That is how you learn how it works. But for every new model you would write a new loop, and choose a learning rate and a number of steps again. To use a model, you usually take a library such as scikit-learn. It has dozens of models, and every one of them works with the same two steps: `fit` to learn from the data, and `predict` to predict. Once you know those, you can swap one model for another by changing a single line.

        **An everyday example.** Every car has a steering wheel and pedals in the same places. Once you can drive one car, you can drive another, even though the engines differ.
        """),
        draw.marked_lines([("Your own code", [("w = np.zeros(X.shape[1])", "1"), ("b = 0.0", ""), ("for step in range(steps):", "2"),
                                              ("    grad_w, grad_b = gradients.mse_gradients(X, y, w, b)", ""),
                                              ("    w = w - lr * grad_w", ""), ("    b = b - lr * grad_b", ""), ("y_pred = X @ w + b", "3")]),
                           ("scikit-learn", [("model = LinearRegression()", "1"), ("model.fit(X, y)", "2"), ("y_pred = model.predict(X)", "3")])],
                          {"1": "make the model", "2": "learn from the data", "3": "predict"},
                          title="Make, fit, predict"),
    ])
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights",
        question="`LinearRegression` from scikit-learn learns from the eight houses of [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en). How do you think its weights compare with what your own `fit` learned?",
        options={
            "same": "About the same",
            "different": "Completely different",
            "lr": "It first needs a learning rate",
        },
        reveal="**About the same:** 204.1 for the area, 16.6 for a garden, −38.8 for the distance, and 77.3 for the bias term. It is the same model with the same loss, so it has the same lowest point. Only the way there differs: scikit-learn works out the lowest point directly, without steps and without a learning rate.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In scikit-learn, a model is something you make, teach and then ask. `LinearRegression()` makes a new model that knows nothing yet. `model.fit(X, y)` lets it learn from the features `X`, a table with one column per feature, and the labels `y`. After that, `model.predict(X_new)` gives one prediction per row of `X_new`, and the learned parameters are in `model.coef_`, the weights, and `model.intercept_`, the bias term. Another model, from the same library, takes exactly the same steps.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **scikit-learn** | | a Python library with many models for machine learning, all with the same steps | `import sklearn` |
        | `fit` | | let a model learn its parameters from the features and the labels | `model.fit(X, y)` |
        | `predict` | | let a trained model predict, one number per row | `model.predict(X_new)` |
        | `coef_`, `intercept_` | coef, intercept | the weights and the bias term a linear model learned; the `_` at the end means "only there after `fit`" | `model.coef_` |
        """),
        L.watch_out("`fit` wants the features as a table, also with a single feature: `df[[\"area\"]]`, with two pairs of brackets. `df[\"area\"]` is one column of numbers, and then `fit` stops with an error."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train and use a model with scikit-learn:

    1. Put the features in a table `X`, one column per feature, and the labels in `y`.
    2. Make the model: `model = LinearRegression()`.
    3. Let it learn: `model.fit(X, y)`.
    4. Predict with `model.predict(X_new)`; the weights are in `model.coef_` and the bias term in `model.intercept_`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The eight houses as a table. `pd.DataFrame` makes a table from a dictionary: each key is a column name, and its list is that column. The columns: area in hundreds of m², garden, distance to the station in km, and the price in thousands of euros. Then a prediction for a new house: 100 m², a garden, 500 m from the station. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_sklearn():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    _df = pd.DataFrame(C.HOUSES)
    # 1. the features as a table, and the labels
    _X = _df[["area", "garden", "distance"]]
    _y = _df["price"]
    # 2. and 3. make the model, and let it learn
    _model = LinearRegression()
    _model.fit(_X, _y)
    print(_model.coef_, _model.intercept_)
    # 4. predict for a new house, also as a table
    _new = pd.DataFrame({"area": [1.0], "garden": [1], "distance": [0.5]})
    print(_model.predict(_new))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The weights are 204.1, 16.6 and −38.8, and the bias term 77.3: the same as your own `fit` reached after 1000 steps. For the new house, the model predicts about 278.6 thousand euros. `predict` gives an array, with one prediction per row of `_new`: here one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many predictions?
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "predict-shape",
        question="`X` has 20 rows and 3 columns. After `model.fit(X, y)`, what does `model.predict(X)` give?",
        options={
            "weights": "3 numbers: the weights",
            "loss": "One number: the loss",
            "right": "20 numbers: one prediction per row",
        },
        correct="right",
        why={
            "weights": "Those are in `model.coef_`. What does `predict` give for each row? See *The idea*.",
            "loss": "`predict` does not compare with the labels. What does it give for each row? See *The idea*.",
            "right": "One prediction for each of the 20 rows; the 3 weights are in `model.coef_`.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Students with scikit-learn

        `df` holds six students: hours of study, hours of sleep, and the grade they got. `y` is the column of grades. Put the features `hours` and `sleep` in `X`, as a table, and replace the second `...` with the two lines that make a `LinearRegression` and let it learn from `X` and `y`. The last line is `model.coef_`, so the check sees the weights. The check also tries other grades.
        """),
        L.task("Train a `LinearRegression` on the students."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-students", lines=10)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit-students", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-students",
        [
            "Steps 1 to 3 of the recipe.",
            "Two columns of a table: how many pairs of square brackets? And the model must exist before it can learn.",
            "`X = df[[\"hours\", ___]]`, then `model = ___()` and `model.___(X, y)`.",
        ],
        why="`df[[\"hours\", \"sleep\"]]` is the table of features. `model = LinearRegression()` makes the model, and `model.fit(X, y)` lets it learn: about 0.42 per hour of study and 0.26 per hour of sleep.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Swap the model

        Is the model any good? Compare it with a model that ignores the features and always predicts the mean grade: `DummyRegressor` from scikit-learn. The box trains the linear model and puts its MSE in `mse_model`. It also makes `mean_model`. Replace the `...` with the same steps for `mean_model`: let it learn, and put the MSE of its predictions in `mse_mean`. The last line shows both. The check also tries other grades.
        """),
        L.task("Put the MSE of the mean model in `mse_mean`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-model", lines=16)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-model", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-model",
        [
            "Look at the lines for `model` just above: the same steps, with the other model.",
            "Which model must learn now, and whose predictions go into `mse_mean`?",
            "`mean_model.fit(___, ___)`, then `mse_mean = ((y - mean_model.___(X)) ** 2).mean()`.",
        ],
        why="`mean_model.fit(X, y)` learns the mean grade, 6.33, and `mean_model.predict(X)` predicts it for every student. Its MSE is about 0.96, against 0.016 for the linear model: the features help a lot.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) you built the model yourself; scikit-learn gives the same model, and many others, behind the same two steps, `fit` and `predict`. The features come in a table, as in [Tables with pandas](/course/open/data-pandas?lang=en). Comparing with a model that always predicts the mean tells you whether a model learned anything at all; *Always beat a simple baseline* comes back to that later, and *Project: predict something real, end to end* uses scikit-learn for a whole project. In the next lesson, *Probability: counting how often*, you start on probability: how often something happens, out of all tries.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_fit, result_mean):
    L.wrap_up(choice_shape, result_fit, result_mean)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
