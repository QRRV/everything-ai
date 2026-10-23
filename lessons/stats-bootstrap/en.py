# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson stats-bootstrap. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("stats-bootstrap", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Random numbers and seeds](/course/open/np-random?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_integers():
    quick_integers = L.choice(
        "quick-integers",
        question="`rng` is a random number generator. Which of these can `rng.integers(0, 3, size=4)` give?",
        options={
            "right": "`[0, 2, 2, 1]`: four numbers from 0 to 2",
            "three": "`[1, 3, 2, 3]`: four numbers from 1 to 3",
            "short": "`[0, 1, 2]`: each number once",
        },
        correct="right",
        why={
            "three": "The second number of `rng.integers` is not included: 3 never comes. See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "short": "`size=4` asks for four numbers, and a number can come more than once. See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "right": "Four whole numbers from 0 up to, but not including, 3; the same number can come more than once.",
        },
    )
    quick_integers
    return (quick_integers,)


@app.cell(hide_code=True)
def quick_check_median():
    quick_median = L.choice(
        "quick-median",
        question="Which number does `np.quantile(x, 0.5)` give?",
        options={
            "mean": "The mean of `x`: the middle",
            "right": "The median of `x`: half of the values lie below it",
            "half": "Half of the largest value of `x`: 0.5 times the top",
        },
        correct="right",
        why={
            "mean": "The mean is the sum divided by the count; a quantile counts values instead. See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "half": "A quantile is a value of `x` below which a share of the values lies, not a fraction of a value. See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "right": "The 0.5 quantile has half of the values below it: the median. This lesson uses the 0.025 and 0.975 quantiles.",
        },
    )
    quick_median
    return (quick_median,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You work out a number from the data you have: a model gets 87 of 100 test examples right, or 68 penguins have a mean bill length of 48.8 mm. On other test examples, or other penguins, the number would come out a little differently. Before you say that one model is better than another, or that a result holds, you want to know how far such a number could move by chance alone. You have only the one set of data, and for most numbers there is no formula.

    **An everyday example.** You taste one spoonful to know how salty a whole pan of soup is. A second spoonful tastes a little different, and how different tells you how far you can trust the first.
    """)
    return


@app.cell(hide_code=True)
def predict_accuracy():
    guess_accuracy = L.guess(
        "accuracy-range",
        question="A model gets 87 of 100 test examples right: an accuracy of 87%. Its true accuracy is the share it would get right of endless examples of the same kind. Between which values do you think that lies?",
        options={
            "close": "From about 86% to 88%: it is the same model",
            "wide": "From about 50% to 100%: 100 examples say almost nothing",
            "right": "From about 80% to 93%: chance moves a few examples",
        },
        reveal="**From about 80% to 93%.** With 100 examples, chance alone moves the count by a few either way. Resample the 100 results 2000 times, as this lesson shows, and the middle 95% of the accuracies runs from 0.80 to 0.93: the 95% confidence interval of the true accuracy. So a second model with 85% on another 100 examples has not been shown to be worse.",
    )
    guess_accuracy
    return (guess_accuracy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A number you work out from data, such as a mean or an accuracy, comes from your **sample**: the examples you happen to have. Another sample of the same kind gives a slightly different number. How much it varies from sample to sample is its **standard error**: the standard deviation the number would have over many samples.

        You have only one sample. The bootstrap of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en) makes new ones from it: draw $n$ values from your $n$ values with replacement, so that a value can come twice or not at all, and work out the number again. Such a **resample** is as large as your data and looks like it, but differs a little, as another sample would. Do that a few thousand times, and the spread of the numbers you get shows how much your number moves by chance.

        A **confidence interval** gives that as a range. The 95% interval of the bootstrap runs from the 0.025 quantile to the 0.975 quantile of the resampled numbers: the middle 95%, with 2.5% cut off at each side. Made this way, the interval catches the true value, the one endless data would give, in about 95 of every 100 samples.

        For a mean there is also a formula: its standard error is about the standard deviation of the data divided by the square root of the number of examples.
        """),
        L.formula(r"\mathrm{SE} \approx \frac{\text{standard deviation of the data}}{\sqrt{n}}"),
        mo.md(r"""
        Four times as many examples halve the standard error, and with it the width of the interval. The bootstrap gives the same without a formula, also for numbers that have none.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **sample** | | the examples you happen to have, out of all there could be | `x` |
        | **standard error**, $\mathrm{SE}$ | standard error | how much a number from a sample varies from sample to sample: its standard deviation over many samples | `means.std()` |
        | **resample** | | one such new sample, as large as the data | |
        | **confidence interval** | | a range around a number that, made this way, catches the true value in a given share of samples, such as 95% | `np.quantile(means, [0.025, 0.975])` |
        | `rng.choice(x, size=n)` | | `n` values drawn from `x` with replacement | |
        """),
        L.watch_out("The bootstrap knows only your sample. If it is small, say 5 values, or not like the rest, say penguins from one island only, the interval is too narrow or in the wrong place, and more resamples do not fix that: only more or better data does. And a 95% interval of a mean does not hold 95% of the values: it is about the mean, not about single penguins."),
        mo.md(r"""
        The picture: the accuracies of 2000 resamples of the 100 test results from the guess. They pile up around 0.87, and the middle 95% runs from 0.80 to 0.93: the 95% confidence interval of the accuracy. It is not quite symmetric: above 0.87 there is less room than below it.
        """),
        mo.Html(C.boot_picture({"title": "2000 resamples of 100 test results", "x": "accuracy of the resample", "y": "resamples",
                                "band": "middle 95%: {low} to {high}", "number": L.number,
                                "alt": "A histogram of 2000 accuracies, from about 0.77 to 0.97, highest around 0.87. Dashed lines at 0.80 and 0.93 mark the middle 95%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the 95% confidence interval of a number with the bootstrap:

    1. Work out the number on your data, for example the mean.
    2. Draw a resample: $n$ values from your data with replacement, and work out the number again.
    3. Repeat step 2 a few thousand times; the standard deviation of the results is the standard error.
    4. Cut 2.5% off each side: the 0.025 and 0.975 quantiles of the results are the 95% confidence interval.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The bill lengths of 68 Chinstrap penguins on Dream, an island of the Palmer Archipelago: `C.chinstrap_bills()` gives them as an array. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_bootstrap():
    import numpy as np

    _x = C.chinstrap_bills()
    # 1. the number: the mean bill length of the 68 penguins
    print(len(_x), _x.mean())
    _rng = np.random.default_rng(0)
    _means = []
    for _i in range(2000):
        # 2. a resample: 68 values drawn with replacement, and its mean
        _resample = _rng.choice(_x, size=len(_x))
        _means.append(_resample.mean())
    _means = np.array(_means)
    # 3. the standard error: the spread of the 2000 means, and the formula next to it
    print(_means.std(), _x.std() / len(_x) ** 0.5)
    # 4. the low end of the 95% interval is here; add the high end, the 0.975 quantile
    print(np.quantile(_means, 0.025))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The 68 penguins have a mean bill length of 48.83 mm. The 2000 resampled means spread by 0.41 mm, close to the formula, 3.31 / $\sqrt{68}$ = 0.40 mm. The low end of the interval is 48.03 mm; with step 4 complete, the high end is 49.61 mm. So the 95% confidence interval of the mean bill length runs from about 48.0 to 49.6 mm, for Chinstrap penguins on Dream, the one island where all 68 were measured. It does not say that the true mean lies in this one range with a chance of 95%: it lies in it or not. It says that intervals made this way catch the true mean in about 95 of every 100 samples.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_four():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Nine times as much data
    """)
    return


@app.cell(hide_code=True)
def choice_four_cell():
    choice_four = L.choice(
        "more-data",
        question="You measure nine times as many Chinstrap penguins: 612 instead of 68. About how wide does the 95% interval of their mean bill length become?",
        options={
            "right": "About a third as wide: the standard error shrinks with the square root of n",
            "ninth": "About a ninth as wide: there is nine times as much data",
            "same": "About as wide: the bootstrap always cuts 2.5% off each side",
        },
        correct="right",
        why={
            "ninth": "The standard error is the standard deviation divided by $\\sqrt{n}$: nine times $n$ divides it by $\\sqrt{9} = 3$, not by 9.",
            "same": "The cut is the same, but the resampled means lie closer together with more data, so the middle 95% is narrower.",
            "right": "The standard error is the standard deviation divided by $\\sqrt{n}$: nine times $n$ divides it by $\\sqrt{9} = 3$. Each extra digit of precision costs a hundred times the data.",
        },
    )
    choice_four
    return (choice_four,)


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The bootstrap means

        Write `bootstrap_means(x, n_boot, seed)`: it makes a random number generator with `seed` and gives back the means of `n_boot` resamples of the array `x`, as an array: steps 2 and 3 of the recipe. The check calls `bootstrap_means(np.array([2, 4, 6, 8]), 1000, 0)`, which should give 1000 means around 5, with a spread of about 1.1, and tries other data. Random numbers come out differently with another seed or other calls, so the check looks at how many means there are, where they lie and how much they spread, and whether the same seed gives the same means.
        """),
        L.task("Make `bootstrap_means` give the means of `n_boot` resamples of `x`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("boot-means", lines=9)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("boot-means", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "boot-means",
        [
            "Steps 2 and 3 of the recipe, as in the worked example: a loop of `n_boot` rounds, each with `rng.choice(x, size=len(x))` and its mean.",
            "Make `rng = np.random.default_rng(seed)` once, before the loop; collect the means in a list with `append`, and return `np.array` of that list.",
        ],
        why="For `[2, 4, 6, 8]`, the means of 1000 resamples lie around 5, the mean of the data, with a spread of about the standard deviation of the data, 2.24, divided by $\\sqrt{4} = 2$: about 1.1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_interval():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The interval

        Write `interval(means, level)`: the confidence interval that holds the share `level` of the means in the middle, as two numbers, low and high, by step 4 of the recipe. For `level` 0.95 those are the 0.025 and 0.975 quantiles. The check calls `interval(np.arange(1, 101), 0.9)`, which should give `[5.95, 95.05]`, and tries other levels.
        """),
        L.task("Make `interval` give the low and high end of the middle share `level` of `means`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_interval():
    box_interval = L.code_box("interval", lines=6)
    box_interval
    return (box_interval,)


@app.cell(hide_code=True)
def check_interval(box_interval):
    result_interval = L.check("interval", box_interval.value)
    result_interval
    return (result_interval,)


@app.cell(hide_code=True)
def hints_interval():
    L.hints(
        "interval",
        [
            "The share outside the interval is `1 - level`; half of it goes to each side.",
            "`np.quantile(means, [low_fraction, high_fraction])` gives both ends at once.",
        ],
        why="For `level` 0.9, 0.1 is left over: 0.05 at each side. The 0.05 and 0.95 quantiles of the numbers 1 to 100 are 5.95 and 95.05.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The standard deviation of [Expected value and variance](/course/open/prob-expectation?lang=en) says how far values lie from their mean; the standard error says how far a mean, or any number from a sample, could lie from the true one. In [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en) the bootstrap gave each tree its own training examples; here the same draws show how much a number varies. The random number generator of [Random numbers and seeds](/course/open/np-random?lang=en) draws the resamples, and the quantiles of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) give the interval. From here on, give every number you measure together with its interval: a difference between two models only counts when it is larger than chance. The bootstrap comes back in *One run is not a result: seeds and variance* and *Is the difference real?*. In the next lesson, *Gradient checking*, you check the gradients of a network against the slope of the loss itself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_four, result_means, result_interval):
    L.wrap_up(choice_four, result_means, result_interval)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
