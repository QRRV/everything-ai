# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-bias-variance. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-bias-variance", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_luck():
    quick_luck = L.choice(
        "quick-luck",
        question="With one validation set of 68 penguins, eight shuffles chose five different values of k. Why?",
        options={
            "learns": "k-NN learns something different each time from the same penguins",
            "right": "Which penguins land in the validation set is luck, and with 68 of them one mistake more or less decides",
            "test": "The test set was different in every shuffle",
        },
        correct="right",
        why={
            "learns": "With the same penguins and the same k, k-NN always predicts the same. What changed between the shuffles? See *Why this matters* in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "test": "The test set stayed the same 68 penguins. See the worked example in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "right": "A few lucky or unlucky penguins decide; cross-validation counts over all of them.",
        },
    )
    quick_luck
    return (quick_luck,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Two bets both pay 5 on average. Bet A pays 4 or 6, bet B pays 0 or 10, each with probability 0.5. Which has the larger variance?",
        options={
            "a": "Bet A",
            "right": "Bet B",
            "same": "Neither: both have an expected value of 5",
        },
        correct="right",
        why={
            "a": "Bet A pays 1 away from 5, bet B 5 away. Which spreads more? See *The idea* in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "same": "The variance measures how far the results spread around the expected value. See *The idea* in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "Bet B: 5² × 0.5 + 5² × 0.5 = 25, against 1 for bet A.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your model is further off on the validation set than you would like. Collect more training examples, which costs time and money, or change the model? That depends on why it is off: because it is too simple for the pattern, or because it follows the chance details of its own training examples. This lesson names the two, and shows a chart that tells them apart.

    **An everyday example.** Two bathroom scales. One always shows 2 kilos too little: ten readings agree, but all are wrong in the same way. The other is right on average, but jumps a kilo up or down every time. Weighing yourself ten times and taking the mean fixes the second scale, never the first.
    """)
    return


@app.cell(hide_code=True)
def predict_extra():
    guess_extra = L.guess(
        "extra-temperatures",
        question="You train a straight line and a curve of degree 9 on 12 temperatures of a day like the one in *Overfitting and underfitting*, and then on 200. For which of the two do you think the extra temperatures make a big difference on new temperatures?",
        options={
            "line": "The straight line",
            "nine": "The curve of degree 9",
            "both": "Both, about the same",
        },
        reveal="**The curve of degree 9.** With 12 temperatures it also follows their noise, and more of them take that away. No number of temperatures teaches a straight line to rise and fall. The worked example below puts numbers on both.",
    )
    guess_extra
    return (guess_extra,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Imagine measuring the same kind of day again and again, and training the same model on every new training set. Its fits, the lines or curves it learns, can then miss in two ways:

        - **Bias**, or in full statistical bias: they miss in the same way on every training set, like a straight line that is too low at midday every time. That is underfitting: more examples do not help, more capacity does.
        - **Variance**, the variance of their predictions over many training sets, as in [Expected value and variance](/course/open/prob-expectation?lang=en): they differ a lot from one training set to the next, because the model also follows the noise of the examples it got. That is overfitting: more examples help, and so does less capacity.

        More capacity lowers the bias and raises the variance: the **bias-variance trade-off**. On new data, a model is off by the noise, its bias and its variance together.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **bias**, statistical bias | | a miss of a model that is the same on every training set | |
        | **variance** of a model | | the variance of its predictions over many training sets: how much they change from one training set to the next | |
        | **bias-variance trade-off** | | more capacity gives less bias and more variance | |
        | **learning curve** | | the training and validation MSE against the number of training examples | |
        """),
        L.watch_out("Bias means more than one thing in this course. Three of them meet here: the **bias term** $b$ of [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en), a parameter; **statistical bias**, a miss that is the same on every training set, as in this lesson; and **data bias**, a skew in a dataset that a model learns, such as groups that are missing or too rare, in the later lesson *Who is in the data, and who is missing?*"),
        mo.md(r"""
        Twenty training sets made by the rule of [Overfitting and underfitting](/course/open/ml-overfitting?lang=en), each with its own hours and noise. On each, a straight line and a curve of degree 9 are trained: the twenty <span class="c-pred">fits</span> of each model in blue, the rule dashed, with a dot at three in the afternoon, where it gives 20 °C. Move the slider to give every set more temperatures.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_sets_cell():
    def _frame(n):
        n = int(n)
        lo1, hi1 = C.spread(1, n)
        lo9, hi9 = C.spread(9, n)
        off = C.far_off(9, n)
        values = {"n": n, "lo1": L.number(lo1, 1), "hi1": L.number(hi1, 1), "lo9": L.number(lo9, 1), "hi9": L.number(hi9, 1), "off": off}
        if off == 20:
            nine = "far apart, and every one leaves the chart"
        elif off == 1:
            nine = "closer around the arch, but one still leaves the chart"
        elif off > 0:
            nine = "closer around the arch, but {off} still leave the chart".format(**values)
        else:
            nine = "close around the arch"
        alt = ("Two charts of the temperature over a day, with the rule as a dashed arch that gives 20 °C at three in the afternoon. "
               "Left, twenty straight lines on training sets of {n} temperatures: they differ from set to set, and each is below the arch at midday, from {lo1} to {hi1} °C at three. "
               "Right, twenty curves of degree 9: ").format(**values) + nine + ", from {lo9} to {hi9} °C at three.".format(**values)
        return (C.picture(n, {"line": "Degree 1", "nine": "Degree 9", "hour": "hour of the day", "temp": "temperature (°C)", "alt": alt}),
                """{n} temperatures per training set. At three o'clock the twenty <span class="c-pred">straight lines</span> say {lo1} to {hi1} °C, the twenty <span class="c-pred">curves of degree 9</span> {lo9} to {hi9} °C; the rule gives 20.""".format(**values))

    explore_sets = L.explore("sets", "n", _frame, start=12, values=C.SIZES, role="",
                             task="From how many temperatures per training set do the twenty curves of degree 9 lie close to the dashed rule? And what do the straight lines do meanwhile?")
    explore_sets
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        With 12 temperatures, the straight lines differ from set to set too, but all miss in the same way: too low at midday, too high early and late. That is bias. The curves of degree 9 go every which way, from 13.0 to 27.3 °C at three o'clock, and every one leaves the chart: variance. From 100 temperatures on, they lie close around the rule. The lines come closer together as well, but with 400 temperatures they still say 17.7 to 18.3 °C at three instead of 20: more examples took their variance away, not their bias.

        ### One training set: the learning curve

        With one training set you cannot see twenty fits, but you can train on more and more of it, its first 12 examples, its first 25, and so on, each time measuring the MSE on those examples and on the validation set. That is the **learning curve**:

        - **High bias**: both MSEs level off close together, far above the noise. More examples will not help; more capacity will.
        - **High variance**: the validation MSE lies far above the training MSE, and keeps falling as the number of examples grows. More examples help; so would less capacity.
        """),
        L.watch_out("A learning curve is not the loss curve of [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en), the loss after every step of training. Some sources call both a learning curve."),
        mo.md(r"""
        The learning curves of the worked example below, both axes on a log scale: the <span class="c-loss">training MSE</span> solid with filled dots, the <span class="c-loss">validation MSE</span> dashed with open dots, and the noise, grey and dashed: an MSE of 0.75, what even the rule itself scores on average.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_curves():
    mo.Html(C.curves_picture({"line": "Degree 1", "nine": "Degree 9", "n": "training temperatures", "mse": "MSE", "noise": "noise", "train": "training set", "val": "validation set",
                              "alt": "Two learning curves, the MSE against the number of training temperatures from 12 to 200, both axes on a log scale. Degree 1: the training MSE stays between 5.1 and 5.7, the validation MSE falls from 6.9 to 5.6; the two end close together, far above the noise of 0.75. Degree 9: the training MSE rises from 0.45 to 0.65, the validation MSE falls from 20.9 to 0.75, down to the noise."}))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To draw and read a learning curve:

    1. Keep a validation set apart, and shuffle the training examples.
    2. For a growing number $n$: train on the first $n$ training examples.
    3. Measure the MSE on those $n$ examples, and on the validation set.
    4. Compare the two against $n$. Close together and high: high bias, so give the model more capacity. Far apart, with the validation MSE still falling: high variance, so more examples help.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three hundred made-up temperatures by the same rule, read at random hours, so already shuffled: the first 200 to train on, the last 100 to validate. For degree 1 and for degree 9, the learning curve at 12, 25, 50, 100 and 200 training temperatures. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_learning_curve():
    import numpy as np

    _rng = np.random.default_rng(75)
    _hours = 24 * _rng.random(size=300)
    _temps = 20 - 0.05 * (_hours - 15) ** 2 + 3 * (_rng.random(size=300) - 0.5)
    # 1. 200 to train on, 100 to validate; the hours are random, so already shuffled
    _x_train, _y_train = _hours[:200], _temps[:200]
    _x_val, _y_val = _hours[200:], _temps[200:]

    for _degree in [1, 9]:
        for _n in [12, 25, 50, 100, 200]:
            # 2. train on the first _n training temperatures
            _w = np.polyfit(_x_train[:_n], _y_train[:_n], _degree)
            # 3. the MSE on those _n, and on the validation set
            _mse_train = np.mean((_y_train[:_n] - np.polyval(_w, _x_train[:_n])) ** 2)
            _mse_val = np.mean((_y_val - np.polyval(_w, _x_val)) ** 2)
            print(_degree, _n, _mse_train, _mse_val)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each line gives the degree, $n$, the training MSE and the validation MSE. Degree 1: the training MSE stays between about 5.1 and 5.7, and the validation MSE falls from about 6.9 to 5.6: close together, far above the noise. High bias, so step 4 says more capacity. Degree 9: about 0.45 against 20.9 with 12 temperatures, 0.65 against 0.75 with 200: the gap closed, as more examples took the variance away. Its training MSE goes up a little, as more temperatures are harder to bend through.

    Another set of temperatures gives other numbers, most of all for degree 9 with few of them; cross-validation at every $n$, as in the lesson before, gives a steadier curve.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Bias or variance?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "bias-or-variance",
        question="Twenty training sets of house prices, and on each the same kind of model. The twenty fits lie almost on top of each other, and every one predicts large houses about 40 thousand euros too cheap. What does this model suffer from most?",
        options={
            "right": "High bias",
            "variance": "High variance",
            "neither": "Neither: the fits agree, so the model is right",
        },
        correct="right",
        why={
            "variance": "Variance shows as fits that differ from one training set to the next. Do they here? See *The idea*.",
            "neither": "Agreeing is not being right: all twenty miss in the same way, like the first bathroom scale.",
            "right": "They miss in the same way on every training set: bias. More houses will not change that; more capacity can.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_curve():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A learning curve

        One hundred and fifty temperatures by the same rule: the first 100 in `x_train` (hours) and `y_train` (temperatures), the last 50 in `x_val` and `y_val`. In the loop, fit a curve of degree `degree` on the first `n` training temperatures, and add its MSE on those `n` to `train_mses`; the box already adds its validation MSE to `val_mses`. Replace the two `...`. The last line is `train_mses, val_mses`, so the check sees both. The check also tries degree 1.
        """),
        L.task("Put the training MSE of a curve fitted on the first `n` training temperatures in `train_mses`, for every `n`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_curve():
    box_curve = L.code_box("learning-curve", lines=16)
    box_curve
    return (box_curve,)


@app.cell(hide_code=True)
def check_curve(box_curve):
    result_curve = L.check("learning-curve", box_curve.value)
    result_curve
    return (result_curve,)


@app.cell(hide_code=True)
def hints_curve():
    L.hints(
        "learning-curve",
        [
            "Steps 2 and 3 of the recipe, as in the worked example, for one degree.",
            "`np.polyfit` gets the first `n` training hours, the first `n` training temperatures and the degree: `[:n]` takes the first `n`. The training MSE is measured on those same `n`.",
            "`w = np.polyfit(x_train[:n], y_train[:n], degree)` and `train_mses.append(np.mean((y_train[:n] - np.polyval(w, ___)) ** 2))`.",
        ],
        why="With degree 6, the training MSEs are about 0.08, 0.37, 0.71 and 0.76, and the validation MSEs about 1.99, 1.35, 0.83 and 0.78: the gap closes as `n` grows.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_more():
    mo.md(r"""
    ### Exercise 3. Take the extra data?
    """)
    return


@app.cell(hide_code=True)
def choice_more_cell():
    choice_more = L.choice(
        "more-data",
        question="A learning curve: with 250, 500 and 1000 training examples, the training MSE is 0.2, 0.3 and 0.4, the validation MSE 6.0, 4.2 and 3.1. The noise is about 0.5. Someone offers you 3000 more examples. Do you take them?",
        options={
            "right": "Yes: high variance, as the validation MSE is far above the training MSE and still falling",
            "worse": "No: the training MSE goes up with more examples, so the model gets worse",
            "capacity": "No: the model is too simple and needs more capacity",
        },
        correct="right",
        why={
            "worse": "More examples are harder to follow, so the training MSE rises; what counts is the validation MSE. See the worked example.",
            "capacity": "A model that is too simple has a high training MSE as well. Is it high here? See *One training set: the learning curve*.",
            "right": "A wide gap that keeps closing: more examples will most likely lower the validation MSE further.",
        },
    )
    choice_more
    return (choice_more,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Overfitting and underfitting](/course/open/ml-overfitting?lang=en) the training and test MSE told underfitting from overfitting; bias and variance say what goes wrong in each, and the learning curve says whether more examples would help. In [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en), which penguins landed in the validation set changed which k won; here, which temperatures land in the training set changes the model itself. In the next lesson, *Regularization and weight decay*, you lower the variance of a flexible model another way: by making large weights cost something.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_curve, choice_more):
    L.wrap_up(choice_which, result_curve, choice_more)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
