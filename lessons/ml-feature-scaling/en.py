# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-feature-scaling. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-feature-scaling", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) and [Broadcasting](/course/open/np-broadcasting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_std():
    quick_std = L.choice(
        "quick-std",
        question="What does the standard deviation of a column of numbers tell you?",
        options={
            "right": "How far the values typically lie from their mean",
            "range": "The largest value minus the smallest",
            "middle": "The middle value, once you sort them",
        },
        correct="right",
        why={
            "range": "That uses only the two outermost values. What does the standard deviation do with all of them? See *The idea* in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "middle": "That is the median. See the table in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "right": "It is the typical distance from the mean, in the unit of the values: small when they lie close together.",
        },
    )
    quick_std
    return (quick_std,)


@app.cell(hide_code=True)
def quick_check_lr():
    quick_lr = L.choice(
        "quick-broadcast",
        question="`X` has shape (8, 3) and `m` has shape (3,). What does `X - m` do?",
        options={
            "error": "Python stops: the shapes do not fit",
            "right": "It takes `m` off every row of `X`: shape (8, 3)",
            "first": "It takes `m` off the first row of `X` only",
        },
        correct="right",
        why={
            "error": "Line the shapes up on the right: 3 and 3 are the same, and the missing length counts as 1. See the rule in [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "first": "NumPy stretches `m` over all 8 rows, without copying it. See the picture in [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "right": "Lined up on the right, (8, 3) and (3,) fit: `m` goes with every row, so each column loses its own number of `m`.",
        },
    )
    quick_lr
    return (quick_lr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) the area was in hundreds of m² and the distance in km, so that all three features had numbers of about the same size. Real data rarely comes like that: an area in m², a distance in meters, an income in euros. One learning rate has to suit every weight, and with features of very different sizes, no learning rate does. You need a way to put every feature on the same scale, whatever its unit.

    **An everyday example.** One test is marked out of 10, another out of 100. Add the two marks, and the second test counts ten times as much. To weigh them fairly, you first put both on the same scale.
    """)
    return


@app.cell(hide_code=True)
def predict_units():
    guess_units = L.guess(
        "units",
        question="The houses of the previous lesson, but with the area in m² and the distance in meters. Gradient descent runs with the same learning rate, 0.2. What do you think happens?",
        options={
            "same": "The same fit as before",
            "slower": "It is slower, but it gets there",
            "blows-up": "The loss blows up",
        },
        reveal="**The training diverges: the loss blows up.** The numbers are now up to 1000 times bigger, and so are the slopes: after 10 steps, the loss is about 3 × 10¹¹⁵. A learning rate small enough for the distance, 0.000001, hardly moves the garden weight and the bias term: after 1000 steps the loss is still about 439, against 6.9 at best.",
    )
    guess_units
    return (guess_units,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The slope for a weight grows with the size of its feature: in `r @ X`, every residual is multiplied by the numbers of that feature's column. Distances in meters give slopes a thousand times bigger than those of a garden of 0 or 1. A step that suits one weight is far too big for another, or far too small: the loss landscape is a long, narrow valley.

        Standardization puts every feature on the same scale: take off its mean, and divide by its standard deviation.
        """),
        L.formula(r"\text{standardized } x = \frac{x - \text{mean of } x}{\text{standard deviation of } x}"),
        mo.md(r"""
        After that, every feature has mean 0 and standard deviation 1, whatever its unit: its numbers lie around 0, most of them between −2 and 2. Taking off the mean matters too: when every area lies far from 0, a bigger $w$ with a smaller $b$ gives almost the same predictions, which is why the valley lies diagonally. The valley becomes round, or nearly so, as features that move together can still stretch it a little, and one learning rate suits every weight. In NumPy, `X.mean(axis=0)` gives one mean per column, and `X - X.mean(axis=0)` takes each one off its own column, as in [Broadcasting](/course/open/np-broadcasting?lang=en).

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **feature scaling** | | bringing features to comparable sizes, so that no feature counts more just because of its unit | |
        | **standardization** | | feature scaling by taking off the mean and dividing by the standard deviation: afterwards, mean 0 and standard deviation 1 | `(X - X.mean(axis=0)) / X.std(axis=0)` |
        """),
        L.watch_out("Standardize a new example with the mean and standard deviation of the training data, not with its own. The model learned its weights on features standardized with those numbers."),
        mo.Html(C.picture({"raw": "Area in m²", "scaled": "Area standardized", "step": "step {k}",
                           "alt": "Two contour plots of the loss over w and b, each with 50 steps of gradient descent from (0, 0). Left, with the area in m²: a long, thin valley lying diagonally; the steps jump back and forth across the valley around w = 3.6, b = 0, and stay far from the lowest point at w = 3.1, b = 46. Right, with the area standardized: round rings around the lowest point at w = 82, b = 306; the steps walk straight to it."})),
        mo.md(r"""
        One feature, the area of the houses in the gradient descent lessons, and the model $w \cdot x + b$. The <span class="c-loss">lines</span> join the places $(w, b)$ with the same loss; the <span class="c-param">points</span> are 50 steps of gradient descent, and the cross is the lowest point. Left, in m², at $\eta$ = 0.00012, about the biggest learning rate that still works there, the steps jump back and forth across the valley while $b$ hardly moves, and they stay far from the cross. Right, standardized, at $\eta$ = 0.3 as in the gradient descent lessons, they reach the cross within about 10 steps.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To train on features with very different units:

    1. Per feature, work out the mean and the standard deviation over the examples: `mean = X.mean(axis=0)` and `std = X.std(axis=0)`.
    2. Standardize: `Z = (X - mean) / std`.
    3. Train on `Z`, with one learning rate for every weight.
    4. For a new example, standardize it with the same `mean` and `std`, and then predict: `(new - mean) / std @ w + b`. `@` between two vectors gives their dot product, one number, as `np.dot` does.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The eight houses in their own units: the area in m², a garden (1 = yes, 0 = no), and the distance to the station in meters. `_fit` is gradient descent as in the previous lesson. First on the raw features, then standardized. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_scaling():
    import numpy as np

    _X = np.array(C.HOUSES_RAW)
    _y = np.array(C.PRICES)

    def _fit(X, lr, steps):
        w = np.zeros(X.shape[1])
        b = 0.0
        for _step in range(steps):
            r = _y - (X @ w + b)
            w = w - lr * -2 * (r @ X) / len(_y)
            b = b - lr * -2 * r.mean()
        return w, b

    def _loss(X, w, b):
        return ((_y - (X @ w + b)) ** 2).mean()

    # without scaling: lr 0.2 blows up, and a tiny lr hardly moves
    _w, _b = _fit(_X, 0.2, 10)
    print(_loss(_X, _w, _b))
    _w, _b = _fit(_X, 0.000001, 1000)
    print(_loss(_X, _w, _b))
    # 1. per feature: the mean and the standard deviation
    _mean = _X.mean(axis=0)
    _std = _X.std(axis=0)
    print(_mean, _std)
    # 2. and 3. standardize, and train with lr 0.2
    _Z = (_X - _mean) / _std
    _w, _b = _fit(_Z, 0.2, 100)
    print(_loss(_Z, _w, _b))
    # 4. a new house: standardize it with the same _mean and _std, then predict
    _new = np.array([120, 0, 700])
    print((_new - _mean) / _std @ _w + _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    On the raw features, $\eta$ = 0.2 gives a loss of about 3 × 10¹¹⁵ after only 10 steps, and $\eta$ = 0.000001 still a loss of about 439 after 1000 steps. The means are 82.6 m², 0.5 and 850 m, the standard deviations about 26.2 m², 0.5 and 384 m. Standardized, the same 0.2 reaches a loss of 6.95 in 100 steps: the best this model can do. A new house of 120 m², without a garden, 700 m from the station, goes through the same `_mean` and `_std`: about 295 thousand euros.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_z():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One standardized number
    """)
    return


@app.cell(hide_code=True)
def choice_z_cell():
    choice_z = L.choice(
        "z-value",
        question="A feature has mean 80 and standard deviation 20. What is the standardized value of 120?",
        options={
            "no-divide": "40",
            "ratio": "1.5",
            "right": "2",
        },
        correct="right",
        why={
            "no-divide": "That is 120 minus the mean. What is the second step of standardization? See the formula in *The idea*.",
            "ratio": "That is 120 divided by the mean. What do you take off first, and what do you divide by? See the formula in *The idea*.",
            "right": "(120 − 80) / 20 = 40 / 20 = 2: two standard deviations above the mean.",
        },
    )
    choice_z
    return (choice_z,)


@app.cell(hide_code=True)
def text_exercise_standardize():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Hours and minutes

        `X` holds four students, one row each: hours of study, and minutes of sleep. Put the standardized `X` in `Z`: every column minus its mean, divided by its standard deviation. Replace the `...`. The last line is `Z`, so the check sees it. The check also tries other students.
        """),
        L.task("Put the standardized `X` in `Z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_standardize():
    box_standardize = L.code_box("standardize", lines=6)
    box_standardize
    return (box_standardize,)


@app.cell(hide_code=True)
def check_standardize(box_standardize):
    result_standardize = L.check("standardize", box_standardize.value)
    result_standardize
    return (result_standardize,)


@app.cell(hide_code=True)
def hints_standardize():
    L.hints(
        "standardize",
        [
            "Steps 1 and 2 of the recipe: the mean and standard deviation of each column, then take off and divide.",
            "Each feature is a column. Which `axis` gives one number per column? And mind which part Python works out first.",
            "`Z = (X - X.mean(axis=___)) / X.std(axis=___)`.",
        ],
        why="`X.mean(axis=0)` and `X.std(axis=0)` give one number per column. `(X - X.mean(axis=0)) / X.std(axis=0)` takes each column's mean off that column and divides by its standard deviation; the brackets make Python subtract first.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A new house

        A model was trained on the standardized houses: `w` and `b` are its weights and bias term. The box has the houses `X`, their `mean` and `std`, and a new house `new`: 100 m², a garden, 500 m from the station. Put the predicted price of `new` in `pred`, in thousands of euros. Replace the `...`. The last line is `pred`, so the check sees it. The check also tries another new house.
        """),
        L.task("Put the predicted price of the new house in `pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict-new", lines=10)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict-new", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict-new",
        [
            "Step 4 of the recipe: the new house goes through the same standardization as the training houses.",
            "Which mean and standard deviation belong to the numbers the model was trained on: those of `new`, or those of `X`?",
            "`pred = (new - ___) / ___ @ w + b`.",
        ],
        why="`(new - mean) / std` standardizes the new house with the numbers of the training houses; `@ w + b` then predicts as always. That gives about 278.6 thousand euros. Without standardizing, the same weights give about −1876.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) the units of the houses were chosen by hand; standardization, with the mean and standard deviation of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en), does it for any data. It makes the loss landscape round, or nearly so, so that one learning rate suits every weight. It comes back in *Project: predict something real, end to end*, and inside neural networks as batch normalization. In the next lesson, *The same model with scikit-learn*, a library fits the same model for you, in a few lines.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_z, result_predict, result_standardize):
    L.wrap_up(choice_z, result_standardize, result_predict)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
