# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson prob-expectation. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-expectation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Probability: counting how often](/course/open/prob-basics?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_die():
    quick_die = L.choice(
        "quick-die",
        question="You throw a fair die once. What is P(6)?",
        options={
            "six": "6",
            "right": "1/6",
            "half": "1/2",
        },
        correct="right",
        why={
            "six": "That is the face, not its probability. A probability lies between 0 and 1. See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "half": "A 6 or not a 6 are two outcomes, but they are not equally likely. See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "right": "Six faces, each equally likely: 1/6 each, about 0.167.",
        },
    )
    quick_die
    return (quick_die,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="Which steps give the standard deviation of a column of numbers?",
        options={
            "right": "The differences from the mean, squared; their mean; and its square root",
            "no-square": "The differences from the mean, and their mean",
            "range": "The largest value minus the smallest, halved",
        },
        correct="right",
        why={
            "no-square": "Those differences add up to 0, so their mean is always 0. What happens to each difference first? See *The idea* in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "range": "That uses only the two outermost values. See *The idea* in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "right": "Squared, so that differences below the mean do not cancel those above; and the square root brings it back to the unit of the values.",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A bet, a game or a model's guess has an outcome that depends on chance. Two questions decide whether it is any good: what does it give on average, if you repeat it many times, and how far do the results swing around that average? Training a model is the same: it makes the loss small on average, over all the examples it could see.

    **An everyday example.** A raffle ticket costs €2, and 1 ticket in 100 wins €150. On average, a ticket gives back €1.50: you lose 50 cents per ticket, even though one buyer in a hundred goes home happy.
    """)
    return


@app.cell(hide_code=True)
def predict_average():
    guess_average = L.guess(
        "average",
        question="You throw a fair die thousands of times. What do you think the average of all the throws comes to?",
        options={
            "three": "3",
            "three-half": "3.5",
            "six": "6",
        },
        reveal="**3.5:** each face 1 to 6 comes up equally often, and their mean is 21 / 6 = 3.5. It is a number the die never shows.",
    )
    guess_average
    return (guess_average,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A number that depends on chance, $X$, such as what a die shows or what a bet pays, has a distribution: each of its values $x_1, x_2, \dots$ with its probability. Its expected value is the average you get in the long run: each value times its probability, added up.
        """),
        L.formula(r"\mathbb{E}[X] = x_1 \cdot P(x_1) + x_2 \cdot P(x_2) + \dots"),
        mo.md(r"""
        The variance measures how far the values spread around the expected value: for each value, its squared distance from the expected value, times its probability, added up.
        """),
        L.formula(r"\mathrm{Var}(X) = (x_1 - \mathbb{E}[X])^2 \cdot P(x_1) + (x_2 - \mathbb{E}[X])^2 \cdot P(x_2) + \dots"),
        mo.md(r"""
        Its square root is the standard deviation of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en), now weighted by the probabilities. It is back in the unit of the values, and says how far they typically lie from the expected value.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **expected value** | | the long-run average of a number that depends on chance: each value times its probability, added up | `(values * probs).sum()` |
        | **variance** | | the squared distance to the expected value, weighted by probability; its square root is the standard deviation | `((values - mean) ** 2 * probs).sum()` |
        | $x_1, x_2, \dots$ | x one, x two, and so on | the values that $X$ can take | |
        | $\mathbb{E}[X]$ | the expected value of X | the expected value of $X$ | |
        | $\mathrm{Var}(X)$ | the variance of X | the variance of $X$ | |
        """),
        L.watch_out("The expected value need not be a value that can happen: a die never shows 3.5. It is also not the most likely value: it is the average over many tries."),
        mo.Html(C.picture({"safe": "A safe bet", "wild": "A wild bet", "x": "what you win", "y": "probability",
                           "alt": "Two bar charts. A safe bet pays 4 or 6, each with probability 0.5. A wild bet pays 0 or 10, each with probability 0.5. In both, a dashed line marks the expected value, 5."})),
        mo.md(r"""
        Both bets give 5 on average: the dashed line. The safe bet pays 4 or 6, each 1 away from 5, so its variance is 1² × 0.5 + 1² × 0.5 = 1. The wild bet pays 0 or 10, each 5 away, so its variance is 5² × 0.5 + 5² × 0.5 = 25. Their standard deviations, 1 and 5, are how far each bet typically pays from 5.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To describe a number that depends on chance:

    1. List every value with its probability: the distribution.
    2. Expected value: multiply each value by its probability, and add up.
    3. Variance: for each value, the squared distance to the expected value, times its probability, added up.
    4. Standard deviation: the square root of the variance.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A fair die: its expected value, variance and standard deviation, and then a check with 6000 throws. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_die():
    import random

    # 1. the distribution: each face with probability 1/6
    _die = {}
    for _face in range(1, 7):
        _die[_face] = 1 / 6
    # 2. the expected value
    _mean = 0
    for _face in _die:
        _mean = _mean + _face * _die[_face]
    print(_mean)
    # 3. the variance
    _var = 0
    for _face in _die:
        _var = _var + (_face - _mean) ** 2 * _die[_face]
    print(_var)
    # 4. the standard deviation
    _std = _var ** 0.5
    print(_std)
    # check: the average of 6000 throws
    random.seed(3)
    _total = 0
    for _i in range(6000):
        _total = _total + random.randint(1, 6)
    print(_total / 6000)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The expected value is 3.5, the variance about 2.92, and the standard deviation about 1.71: a throw typically lies about 1.7 from 3.5. The average of 6000 throws with `random.randint` is 3.48, close to 3.5. The more throws, the closer it gets.
    """)
    return


@app.cell(hide_code=True)
def text_numpy():
    mo.md(r"""
    **The same in NumPy.** Put the values in one array and their probabilities in another, in the same order. Then each step of the recipe is one line, without a loop:

    | Step | Plain Python | NumPy |
    |---|---|---|
    | 1 | a dictionary `_die` | two arrays, `_faces` and `_probs` |
    | 2 | a loop that adds `_face * _die[_face]` | `(_faces * _probs).sum()` |
    | 3 | a loop that adds `(_face - _mean) ** 2 * _die[_face]` | `((_faces - _mean) ** 2 * _probs).sum()` |
    """)
    return


@app.cell
def example_die_numpy():
    import numpy as np

    _faces = np.array([1, 2, 3, 4, 5, 6])          # 1. the values
    _probs = np.array([1, 1, 1, 1, 1, 1]) / 6      # 1. their probabilities, in the same order
    _mean = (_faces * _probs).sum()                # 2. the expected value
    _var = ((_faces - _mean) ** 2 * _probs).sum()  # 3. the variance
    print(_mean, _var, _var ** 0.5)                # 2, 3 and 4 at once
    return


@app.cell(hide_code=True)
def text_exercise_game():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Worth playing?
    """)
    return


@app.cell(hide_code=True)
def choice_game_cell():
    choice_game = L.choice(
        "game",
        question="A game: you win €10 with probability 0.2, and otherwise nothing. What is the expected value of what you win?",
        options={
            "prize": "€10",
            "half": "€5",
            "right": "€2",
        },
        correct="right",
        why={
            "prize": "That is what you win when you win. How often is that? See the formula in *The idea*.",
            "half": "That is the plain mean of €0 and €10, as if both were equally likely. See the watch-out and step 2 of the recipe.",
            "right": "10 × 0.2 + 0 × 0.8 = 2: on average €2 per game. If a game costs more than €2, you lose in the long run.",
        },
    )
    choice_game
    return (choice_game,)


@app.cell(hide_code=True)
def text_exercise_expected():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The expected value as a function

        Finish `expected(dist)`. It gets a distribution as a dictionary, each value with its probability, and gives back the expected value. Replace the `...`. The check calls `expected({1: 0.5, 2: 0.25, 4: 0.25})`: that should give `2.0`. It also tries other distributions, such as a fair die.
        """),
        L.task("Make `expected` give back the expected value of `dist`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_expected():
    box_expected = L.code_box("expected", lines=7)
    box_expected
    return (box_expected,)


@app.cell(hide_code=True)
def check_expected(box_expected):
    result_expected = L.check("expected", box_expected.value)
    result_expected
    return (result_expected,)


@app.cell(hide_code=True)
def hints_expected():
    L.hints(
        "expected",
        [
            "Step 2 of the recipe, as in the worked example.",
            "A loop over the keys of `dist` gives each value; `dist[value]` is its probability. What do you add to the total for each one?",
            "`total = total + value * ___`, and after the loop `return total`.",
        ],
        why="The loop adds `value * dist[value]` for every value: 1 × 0.5 + 2 × 0.25 + 4 × 0.25 = 2.0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_variance():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How wild is the prize?

        A prize draw pays €0, €5 or €20. `values` holds the prizes and `probs` their probabilities, as NumPy arrays in the same order, and `mean` is the expected value, worked out for you. Put the variance in `variance`. Replace the `...`. The last line is `variance`, so the check sees it. The check also tries other probabilities.
        """),
        L.task("Put the variance of the prize in `variance`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_variance():
    box_variance = L.code_box("variance", lines=8)
    box_variance
    return (box_variance,)


@app.cell(hide_code=True)
def check_variance(box_variance):
    result_variance = L.check("variance", box_variance.value)
    result_variance
    return (result_variance,)


@app.cell(hide_code=True)
def hints_variance():
    L.hints(
        "variance",
        [
            "Step 3 of the recipe, with arrays: NumPy does every value at once.",
            "`values - mean` gives the distance of each value to the expected value. What happens to those distances, and what do you multiply them by?",
            "`variance = ((values - mean) ** 2 * ___).sum()`.",
        ],
        why="`(values - mean) ** 2` squares each distance to 5.5; `* probs` weights it by its probability; `.sum()` adds up: 30.25 × 0.5 + 0.25 × 0.3 + 210.25 × 0.2 = 57.25.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Probability: counting how often](/course/open/prob-basics?lang=en) you made distributions; the expected value is their long-run average, and the variance measures how far results spread around it. Its square root is the standard deviation of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en), weighted by the probabilities. Later in the course, a loss is an expected value: the average over all examples a model could see. In the next lesson, *Kinds of learning*, you tell the kinds of learning apart by where the right answers come from. With *Show extra lessons* on, *Bayes' rule* comes first: it turns a conditional probability around.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_game, result_expected, result_variance):
    L.wrap_up(choice_game, result_expected, result_variance)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
