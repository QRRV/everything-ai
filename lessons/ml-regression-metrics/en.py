# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-regression-metrics. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-regression-metrics", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Always beat a simple baseline](/course/open/ml-baselines?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="A model predicts the prices of 8 test houses with an MSE of 3500. Always predicting the mean training price gets an MSE of 3088 on the same houses. What does that tell you?",
        options={
            "good": "Not much: an MSE of 3500 is normal for house prices",
            "right": "The model does not beat the baseline, so it has not shown it learned anything from the features",
            "compare": "Nothing: an MSE cannot be compared with a baseline",
        },
        correct="right",
        why={
            "good": "An MSE has no fixed scale; what gives it a meaning? See *The idea* in [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "compare": "Measure both on the same test set with the same metric, and you can compare them. See the recipe in [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "right": "The simplest rule, which ignores the features, is closer: 3088 against 3500.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="The values 1, 3 and 5 are equally likely. What is their variance?",
        options={
            "right": "About 2.67",
            "std": "About 1.63",
            "mean": "3",
        },
        correct="right",
        why={
            "std": "That is the standard deviation, the square root of the variance. See the recipe in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "mean": "That is the expected value. The variance is about the squared distances to it. See the recipe in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "The values are 2, 0 and 2 away from 3: (4 + 0 + 4) / 3 is about 2.67.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Always beat a simple baseline](/course/open/ml-baselines?lang=en) the line had an MSE of 648 on its 8 test houses, against 3088 for the baseline: it learned something from the area. But someone who sells a house asks how far off the price will be, and 648 is in thousand euros squared, a unit nobody thinks in. You want a number in euros, and one without a unit that says how much of the differences between prices the model explains.

    **An everyday example.** A weather app that says its forecasts have a mean squared miss of 4 degrees squared tells you little. "Usually about 2 degrees off" tells you what to expect.
    """)
    return


@app.cell(hide_code=True)
def predict_typical():
    guess_typical = L.guess(
        "typical-miss",
        question="The line's MSE on the 8 test houses is 648. About how far off do you think it is on a typical test house?",
        options={
            "mse": "About 648 thousand euros",
            "root": "About 25 thousand euros",
            "tiny": "About 6 thousand euros",
        },
        reveal="**About 25 thousand euros.** The MSE is in thousand euros squared. Its square root, the RMSE below, is about 25.5 thousand euros, and the mean size of the eight residuals, the MAE, about 22.6.",
    )
    guess_typical
    return (guess_typical,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Three metrics for a model that predicts numbers, next to the MSE:

        - The **mean absolute error**, MAE: the mean size of the residuals, without their sign. In the unit of the label: on average, this far off.
        - The **root mean squared error**, RMSE: the square root of the MSE. Also in the unit of the label, but large misses count more, as each residual is squared first. It is never below the MAE; the further above it, the more uneven the misses.
        - **R²**, the coefficient of determination: 1 minus the MSE divided by the variance of the labels. That variance, from [Expected value and variance](/course/open/prob-expectation?lang=en), is the MSE of always predicting the mean of those labels, so R² is the share of it the model explains: 1 is perfect, 0 no better than the mean, below 0 worse. It has no unit.
        """),
        L.formula(r"[[loss:\text{MAE}]] = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - [[pred:\hat{y}_i]] \rvert \qquad [[loss:\text{RMSE}]] = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - [[pred:\hat{y}_i]])^2}"),
        L.formula(r"R^2 = 1 - \frac{[[loss:\text{MSE}]]}{\mathrm{Var}(y)}"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **mean absolute error**, MAE | | the mean of the residuals without their sign | `np.mean(abs(residuals))` |
        | **root mean squared error**, RMSE | | the square root of the MSE: in the unit of the label, with large misses counting more | `math.sqrt(mse)` |
        | **R²**, coefficient of determination | R squared | 1 minus the MSE divided by the variance of the labels: the share of that variance the model explains | `1 - mse / variance` |
        | $\lvert x \rvert$ | the absolute value of x | $x$ without its sign: $\lvert -3 \rvert = 3$. The `abs` of [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en) also does it for every number of an array | `abs(x)` |
        """),
        L.watch_out("R² compares with the mean of the very labels it is worked out on, here the 8 test prices. Nobody could know that mean in advance, so it is not the baseline of [Always beat a simple baseline](/course/open/ml-baselines?lang=en), the mean training price. It is the best single prediction for exactly these prices, so it misses by less than that baseline, and R² is stricter: about 0.65, where 1 − 648 / 3088 gives 0.79."),
        mo.md(r"""
        The 8 test houses of [Always beat a simple baseline](/course/open/ml-baselines?lang=en), with the <span class="c-pred">line</span> learned from the training houses and every <span class="c-residual">residual</span> as a red stick. The house closest to the line now sells for less, say to a family member. Move the slider, or drag across the chart of the MAE and the RMSE, to make it cheaper; that chart shows the <span class="c-loss">MAE</span> (solid) and the <span class="c-loss">RMSE</span> (dashed) for every price of that house.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_cheaper_cell():
    def _frame(d):
        d = int(d)
        m = C.metrics(d)
        values = {"d": d, "r": L.number(m["residual"], 1), "mae": L.number(m["mae"], 1), "rmse": L.number(m["rmse"], 1), "r2": L.number(m["r2"], 2)}
        if d == 0:
            house = "The house closest to the line, at its price in the data: its {role} is {r} thousand euros."
        else:
            house = "The house closest to the line, {d} thousand euros cheaper: its {role} is {r} thousand euros."
        alt = ("Left, eight test houses by area and price, a rising line, and a red stick from each house to the line. " + house.format(role="residual", **values) +
               " Right, the MAE and the RMSE for every price of that house: both rise as it gets cheaper, the RMSE faster once the house is far off, from 25.5 to 74.3 against 22.6 to 46.9 for the MAE; "
               "now the MAE is {mae} and the RMSE {rmse} thousand euros.".format(**values))
        return (C.picture(d, {"houses": "Test houses", "x": "area (hundreds of m²)", "y": "price (thousand euros)", "metrics": "MAE and RMSE",
                              "cheaper": "how much cheaper (thousand euros)", "euros": "thousand euros", "alt": alt}),
                house.format(role="""<span class="c-residual">residual</span>""", **values)
                + """ <span class="c-loss">MAE</span> {mae}, <span class="c-loss">RMSE</span> {rmse} thousand euros; R² {r2}.""".format(**values))

    explore_cheaper = L.explore("cheaper", "cheaper", _frame, start=0, low=0, high=200, step=10, role="",
                                task="Make the house 100 thousand euros cheaper. How much do the MAE and the RMSE go up?")
    explore_cheaper
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 100 thousand euros cheaper, the MAE goes from 22.6 to 34.4, up 11.8: one eighth of the 94.7 thousand euros the house is now further off, as it is one house of eight. The RMSE goes from 25.5 to 42.8, up 17.3, and at 200 thousand cheaper it is 74.3, against 46.9 for the MAE. One large miss weighs more and more, because it is squared. R² is 0.19 at 200 thousand cheaper, against 0.65 at the start.

    So which one do you report?

    - The **MAE** when every euro off costs the same, and a few odd examples should not decide.
    - The **RMSE** when large misses are much worse than small ones. It also goes with the MSE you train on: lowering one lowers the other.
    - **R²** to say how much of the spread a model explains, as one number without a unit. Compare it only on the same labels: with the same misses, labels that spread more give a higher R². So put the MAE or the RMSE next to it.

    ### Recipe

    To judge a model that predicts numbers:

    1. Work out the residuals on the test set: `y - y_pred`.
    2. MAE: the mean of `abs(residuals)`.
    3. RMSE: the square root of the mean of `residuals ** 2`, with `math.sqrt`.
    4. R²: 1 minus the MSE divided by the variance of the labels, the mean of `(y - y.mean()) ** 2`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The 8 test houses of [Always beat a simple baseline](/course/open/ml-baselines?lang=en), and the line learned from its 32 training houses, prices in thousand euros. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_metrics():
    import math
    import numpy as np

    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)
    _order = _rng.permutation(40)
    _x_test, _y_test = _sizes[_order[32:]], _prices[_order[32:]]
    _w, _b = 152.82, 52.32   # the line learned from the training houses

    # 1. the residuals on the test set
    _residuals = _y_test - (_w * _x_test + _b)
    print(_residuals)
    # 2. MAE: the mean size of the residuals
    print(np.mean(abs(_residuals)))
    # 3. RMSE: the square root of the MSE
    _mse = np.mean(_residuals ** 2)
    print(_mse, math.sqrt(_mse))
    # 4. R²: 1 minus the MSE divided by the variance of the test prices
    _variance = np.mean((_y_test - _y_test.mean()) ** 2)
    print(_variance, 1 - _mse / _variance)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The residuals run from about −10.1 to 39.4 thousand euros, and seven of the eight are positive: these houses happen to cost more than the line predicts. The MAE is about 22.6 thousand euros: on average the line is this far off. The RMSE is about 25.5, only a little higher, as no single miss stands out. The variance of the 8 prices is about 1833, so R² is 1 − 648 / 1833, about 0.65: the line explains about two thirds of how far these prices spread.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mae():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. MAE and RMSE

        A weather forecast for six days: the temperatures in `y`, what was forecast in `y_pred`, both in degrees, and the residuals in `residuals`. Put the MAE in `mae`, and the RMSE in `rmse`. Replace the two `...`. The last line is `mae, rmse`, so the check sees both. The check also tries another forecast.
        """),
        L.task("Put the MAE of the forecast in `mae`, and its RMSE in `rmse`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mae():
    box_mae = L.code_box("mae-rmse", lines=9)
    box_mae
    return (box_mae,)


@app.cell(hide_code=True)
def check_mae(box_mae):
    result_mae = L.check("mae-rmse", box_mae.value)
    result_mae
    return (result_mae,)


@app.cell(hide_code=True)
def hints_mae():
    L.hints(
        "mae-rmse",
        [
            "Steps 2 and 3 of the recipe.",
            "`abs` takes away the sign of every number of an array, and `np.mean` takes the mean. For the RMSE: first the mean of the squares, then one `math.sqrt`.",
            "`mae = np.mean(abs(___))` and `rmse = math.sqrt(np.mean(residuals ** ___))`.",
        ],
        why="The residuals are −1, 1, 0, −3, 0 and 6. Their sizes add up to 11, so the MAE is 11 / 6, about 1.83. Their squares add up to 47, so the MSE is 47 / 6, about 7.83, and the RMSE about 2.80.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_r2():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. R²

        The same forecast; `mse` already holds its MSE. Put the variance of the six temperatures in `variance`, and R² in `r2`. Replace the two `...`. The last line is `r2`, so the check sees it. The check also tries another forecast.
        """),
        L.task("Put the R² of the forecast in `r2`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_r2():
    box_r2 = L.code_box("r2", lines=8)
    box_r2
    return (box_r2,)


@app.cell(hide_code=True)
def check_r2(box_r2):
    result_r2 = L.check("r2", box_r2.value)
    result_r2
    return (result_r2,)


@app.cell(hide_code=True)
def hints_r2():
    L.hints(
        "r2",
        [
            "Step 4 of the recipe.",
            "The variance is the mean of the squared distances of `y` to its own mean, `y.mean()`. R² is 1 minus `mse` divided by it.",
            "`variance = np.mean((y - ___) ** 2)` and `r2 = 1 - ___ / variance`.",
        ],
        why="The six temperatures have a mean of 13, and their squared distances to it add up to 56, so the variance is 56 / 6, about 9.33. R² is 1 − 7.83 / 9.33, about 0.161.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 3. Which one to report?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-metric",
        question="A delivery service predicts when each parcel will arrive. One prediction that is an hour off angers a customer far more than two predictions that are each half an hour off. Which metric fits best to judge the predictions?",
        options={
            "mae": "The MAE: every minute off counts the same",
            "right": "The RMSE: large misses count more",
            "r2": "R²: it has no unit",
        },
        correct="right",
        why={
            "mae": "Here a miss of an hour is worse than two of half an hour, and the MAE counts them the same. See *So which one do you report?*",
            "r2": "R² says how much of the spread the model explains, not how bad its large misses are. See *So which one do you report?*",
            "right": "The RMSE squares each miss first, so one hour off weighs more than two half hours, as it does for the customers.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The MSE of [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en) is still what a model trains on; as [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en) said, the number you report can be another one. In [Always beat a simple baseline](/course/open/ml-baselines?lang=en) the mean training price was the baseline; R² measures against the mean of the labels themselves. In the next lesson, *Project: a linear classifier for handwritten digits*, you train a model on real images, and the first thing to check is whether it beats always guessing the most common digit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_mae, result_r2, choice_which):
    L.wrap_up(result_mae, result_r2, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
