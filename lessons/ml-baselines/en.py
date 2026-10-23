# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-baselines. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-baselines", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_accuracy():
    quick_accuracy = L.choice(
        "quick-accuracy",
        question="A model gives three photos the probabilities `[0.2, 0.8]`, `[0.6, 0.4]` and `[0.3, 0.7]`, and all three have the label 1. What is its accuracy?",
        options={
            "mean": "About 0.63, the mean probability of class 1",
            "one": "1, because all three labels are the same",
            "right": "About 0.67",
        },
        correct="right",
        why={
            "mean": "That is how sure the model is of class 1 on average. Which class does it predict for each photo? See the recipe in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "one": "Accuracy compares each prediction with its label. What does the model predict for the second photo? See the recipe in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "right": "The most likely classes are 1, 0 and 1: two of the three are right.",
        },
    )
    quick_accuracy
    return (quick_accuracy,)


@app.cell(hide_code=True)
def quick_check_train():
    quick_train = L.choice(
        "quick-train",
        question="Which number does gradient descent train on, and which do you report?",
        options={
            "right": "It trains on the loss, and you report a metric such as accuracy",
            "metric": "It trains on the accuracy, and you report the loss",
            "same": "Both are the same number",
        },
        correct="right",
        why={
            "metric": "Accuracy is a staircase without slopes. See *The idea* in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "same": "They are different numbers that can move apart. See *The idea* in [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en).",
            "right": "The loss has slopes to follow; the metric says what people care about.",
        },
    )
    quick_train
    return (quick_train,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model gets an accuracy of 0.90 on its test set. Is that good? On its own, the number says nothing: if nine of ten examples belong to one class, always guessing that class also gets 0.90. Before you trust a score, compare it with the simplest prediction you can make without learning anything from the features. If the model does not beat that, its score shows nothing it has learned.

    **An everyday example.** A weather forecast of "tomorrow it will be like today" is right surprisingly often. A forecasting service is only worth paying for if it does better than that.
    """)
    return


@app.cell(hide_code=True)
def predict_good():
    guess_good = L.guess(
        "good",
        question="A line predicts the prices of 8 test houses with an MSE of about 648. What do you think predicting the mean training price for every house gets on the same houses?",
        options={
            "lower": "Less than 648: the mean is the best single prediction",
            "same": "About the same: the houses lie close to the line anyway",
            "higher": "Much more, several times 648",
        },
        reveal="**Much more: about 3088**, almost five times the MSE of the line. The mean training price, about 248 thousand euros, is far off for the cheap and the expensive houses, so the line did learn something from the area. Without the 3088 next to it, the 648 says little: an MSE is in thousand euros squared and has no fixed scale.",
    )
    guess_good
    return (guess_good,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **baseline** is a simple prediction to compare a model with, such as a rule that ignores the features. The usual ones:

        - for classification: always predict the **most common class** of the training set;
        - for regression: always predict the **mean label** of the training set.

        The baseline learns only from the training set, like the model, and you measure both on the same test set with the same metric. A model that does not beat its baseline has not shown, on that metric, that it learned anything from the features, however good its score looks. The bigger the gap, the more the model adds. While you are still choosing, compare with the baseline on the validation set; at the end, report both on the test set.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **baseline** | | a simple prediction to compare a model with, such as always the most common class or the mean label of the training set | `counts.argmax()`, `y_train.mean()` |
        """),
        L.watch_out("Take the baseline from the training set, never from the test set. The mean of the test labels is the best single prediction for exactly those examples, so it beats any number you could have picked before seeing them: that is not a fair baseline."),
        mo.md(r"""
        The houses from [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en): 32 training houses (filled) and 8 test houses (open). The solid <span class="c-pred">line</span> was learned from the training houses; the dashed line is the baseline, the mean training price for every house.
        """),
        mo.Html(C.picture({"title": "A line and its baseline", "x": "area (hundreds of m²)", "y": "price (thousand euros)", "baseline": "baseline",
                           "alt": "Forty houses by area and price, 32 filled and 8 open. A rising straight line goes through them; a flat dashed line at about 248 thousand euros is the baseline, far from the cheap and the expensive houses."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To judge a score:

    1. Pick a baseline: the most common class, or the mean label, of the training set.
    2. Measure the baseline on the test set with the same metric as the model.
    3. Compare: only the gap between the model and its baseline shows what the model learned.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Both cases of the recipe. First classification: ten training mails and ten test mails, where 1 is spam, and the classes a spam filter predicts for the test mails. Then regression: the houses in the picture, with the line learned from the training houses. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_baselines():
    import numpy as np

    # classification: ten training mails and ten test mails, 1 is spam
    _mails_train = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    _mails_test = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    _predicted = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])   # what the filter predicts for the test mails
    # 1. the baseline: the most common class of the training set
    _counts = np.eye(2)[_mails_train].sum(axis=0)
    _most_common = _counts.argmax()
    print(_counts, _most_common)
    # 2. both on the same test set, with accuracy
    print(np.mean(_predicted == _mails_test), np.mean(_mails_test == _most_common))

    # regression: the houses in the picture
    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)
    _order = _rng.permutation(40)
    _x_train, _y_train = _sizes[_order[:32]], _prices[_order[:32]]
    _x_test, _y_test = _sizes[_order[32:]], _prices[_order[32:]]
    _w, _b = 152.82, 52.32   # the line learned from the training houses

    # 1. the baseline: the mean training price
    _baseline = _y_train.mean()
    print(_baseline)
    # 2. both on the same test set, with the MSE
    print(np.mean((_y_test - (_w * _x_test + _b)) ** 2))
    print(np.mean((_y_test - _baseline) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Eight of the ten training mails are not spam, so the baseline always predicts class 0. On the test mails, the filter gets an accuracy of 0.8, and so does the baseline: the 0.8 of the filter shows nothing it has learned. The mean training price is about 248 thousand euros. On the 8 test houses, the line has an MSE of about 648, the baseline about 3088: almost five times as much, so the area really helps to predict the price. Step 3 is the comparison itself.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_disease():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A rare disease
    """)
    return


@app.cell(hide_code=True)
def choice_disease_cell():
    choice_disease = L.choice(
        "rare-disease",
        question="Two in a hundred people have a disease. A model that should spot it gets an accuracy of 0.97 on the test set. What does that tell you?",
        options={
            "good": "It works well: 97 of 100 right",
            "right": "It is worse than always saying 'healthy', which gets 0.98",
            "nothing": "Nothing: accuracy cannot be used for a disease",
        },
        correct="right",
        why={
            "good": "What does the simplest rule get here, always the most common class? See *The idea*.",
            "nothing": "Accuracy can be used; it just needs something to compare with. See *The idea*.",
            "right": "Always predicting the most common class, healthy, gets 98 of 100 right. The model does not even beat that, so its 0.97 shows nothing it has learned. Whether it finds the sick people, accuracy does not tell you.",
        },
    )
    choice_disease
    return (choice_disease,)


@app.cell(hide_code=True)
def text_exercise_majority():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The most common class

        Twelve training labels and eight test labels, of three classes. `counts` already holds how often each class occurs in the training set, with one-hot vectors as in [One-hot vectors for classes](/course/open/ml-one-hot?lang=en). Put the most common class in `most_common`, and the accuracy on the test set of always predicting it in `accuracy`. Replace the two `...`. The last line is `accuracy`, so the check sees it. The check also tries other training labels.
        """),
        L.task("Put the test accuracy of always predicting the most common training class in `accuracy`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_majority():
    box_majority = L.code_box("majority", lines=8)
    box_majority
    return (box_majority,)


@app.cell(hide_code=True)
def check_majority(box_majority):
    result_majority = L.check("majority", box_majority.value)
    result_majority
    return (result_majority,)


@app.cell(hide_code=True)
def hints_majority():
    L.hints(
        "majority",
        [
            "Steps 1 and 2 of the recipe for classification.",
            "Which method gives the place of the largest number in `counts`? Then compare every test label with that class, as for accuracy.",
            "`most_common = counts.___()` and `accuracy = np.mean(y_test == ___)`.",
        ],
        why="The counts are 3, 7 and 2, so class 1 is the most common. Of the 8 test labels, 5 are 1: an accuracy of 0.625.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The mean price

        Six training prices and three test prices. Put the mean training price in `baseline`, and the MSE on the test set of predicting it for every house in `mse`. Replace the two `...`. The last line is `mse`, so the check sees it. The check also tries other training prices.
        """),
        L.task("Put the test MSE of always predicting the mean training price in `mse`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-baseline", lines=7)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-baseline", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-baseline",
        [
            "Steps 1 and 2 of the recipe for regression.",
            "Which set gives the baseline, and on which set do you measure it? The prediction is the same number for every house.",
            "`baseline = ___.mean()` and `mse = np.mean((___ - baseline) ** 2)`.",
        ],
        why="The mean training price is 240. The test prices are 20, 40 and 50 away from it; their squares, 400, 1600 and 2500, have a mean of 1500.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en) you chose the metric to report; a baseline gives that number a meaning. In [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en) the line was measured on new houses; now you also know it beats the simplest rule. In the next lesson, *How far off? MAE, RMSE and R²*, you turn the MSE of the line into a number in euros, and its comparison with a mean into one number without a unit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_disease, result_majority, result_mean):
    L.wrap_up(choice_disease, result_majority, result_mean)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
