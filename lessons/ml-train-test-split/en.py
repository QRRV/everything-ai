# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-train-test-split. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-train-test-split", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) and [Random numbers and seeds](/course/open/np-random?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parameters():
    quick_parameters = L.choice(
        "quick-parameters",
        question="`X` has the shape (50, 3): 50 houses with 3 features each. How many numbers does linear regression learn?",
        options={
            "weights": "3",
            "right": "4",
            "all": "150",
        },
        correct="right",
        why={
            "weights": "That is one weight per feature. What else does the model add? See *The idea* in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "all": "The model learns per feature, not per house. See *The idea* in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "right": "One weight per feature, 3, plus the bias term $b$.",
        },
    )
    quick_parameters
    return (quick_parameters,)


@app.cell(hide_code=True)
def quick_check_permutation():
    quick_permutation = L.choice(
        "quick-permutation",
        question="`rng = np.random.default_rng(0)`. What does `rng.permutation([120, 250, 310])` give?",
        options={
            "one": "One of the three numbers, picked at random",
            "new": "Three random numbers between 0 and 1",
            "right": "The three numbers in a random order, each once",
        },
        correct="right",
        why={
            "one": "`permutation` keeps every item; it only changes their order. See the table in [Random numbers and seeds](/course/open/np-random?lang=en).",
            "new": "That is what `rng.random(size=3)` gives. See the table in [Random numbers and seeds](/course/open/np-random?lang=en).",
            "right": "`permutation` gives back every item once, in a random order, as a new array.",
        },
    )
    quick_permutation
    return (quick_permutation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You train a model on examples whose answers you know, but you need it for examples whose answers you do not know yet: the price of a house that comes on the market tomorrow. A low loss on the training examples does not tell you how it does on those. To find out, you hold back some examples with known answers, train on the rest, and measure on the ones the model never saw.

    **An everyday example.** A cook who only lets the people who helped make the soup taste it hears that it is perfect. To know whether the soup is good, you ask someone who was not in the kitchen.
    """)
    return


@app.cell(hide_code=True)
def predict_look_up():
    guess_look_up = L.guess(
        "look-up",
        question="Two models learn from the same 32 houses. A line gets a mean squared error of 482 on them. A look-up model, which remembers every house it learned from, gets 0. Which do you think predicts 8 other houses better?",
        options={
            "look-up": "The look-up model: on its houses it is never off",
            "line": "The line",
            "same": "Both about the same",
        },
        reveal="**The line.** On the 8 other houses, the line gets an MSE of 648 and the look-up model 1863. The look-up model gives a new house the price of the training house with the closest area, so it is exactly right on houses it has seen and repeats their chance ups and downs on every new one. Its 0 says nothing about new houses.",
    )
    guess_look_up
    return (guess_look_up,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A model is useful when it works on examples it has not seen: that is called **generalization**. You cannot measure it on the examples the model learned from, so you split the examples whose answers you know in two before training. The **training set**, often about 80%, is what the model learns from. The **test set**, the rest, stays out of sight until the end. Then you measure the loss on the test set once: that is your estimate of how the model does on new data.

        Shuffle before you split. Data often comes sorted, by area, by price or by class, and then the last 20% is not like the rest. With `rng.permutation(n)` you shuffle the positions 0 to $n - 1$; the first 80% of them go to training, the rest to testing. An array of positions between square brackets selects the elements at those positions, in that order: `sizes[train]`.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **generalization** | | how well a trained model works on new data it did not see while training | |
        | **training set** | | the examples the model is trained on | `x_train`, `y_train` |
        | **test set** | | examples the model never saw while training, used once at the end to measure how well it works on new data | `x_test`, `y_test` |
        | **`rng.permutation(n)`** | | with a whole number $n$: the positions 0 to $n - 1$ in a random order, each once; the same as `rng.permutation(np.arange(n))` | `rng.permutation(4)` |
        | **elements by position** | | an array of positions between square brackets selects those elements, in that order | `sizes[np.array([2, 0])]` |
        """),
        L.watch_out("Use the test set once, at the end. If you try several models and keep the one with the best test score, the test set has helped you choose, and its score is no longer an honest estimate for new data. A later lesson shows how to choose without it."),
        mo.md(r"""
        The picture shows the forty houses of the worked example below. The filled dots are the training set, the open dots the test set. One chart has a <span class="c-pred">line</span> learned from the training houses; the other the <span class="c-pred">look-up model</span>, which gives every area the price of the training house closest to it. Move the slider for another shuffle.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_shuffle_cell():
    def _frame(k):
        k = int(k)
        e = {key: L.number(v, 0) for key, v in C.errors(k).items()}
        return (C.picture(k, {"line": "A line", "look": "A look-up model", "x": "area (hundreds of m²)", "y": "price (thousand euros)",
                              "alt": f"Two charts of forty houses by area and price, 32 filled training houses and 8 open test houses. One has a straight line through them, the other a staircase that jumps to the price of each training house. The line has a test MSE of {e['line_test']}, the look-up model {e['look_test']}."}),
                f"""Shuffle {k}. The line: an MSE of <span class="c-loss ea-num">{e['line_train']}</span> on the training set and <span class="c-loss ea-num">{e['line_test']}</span> on the test set. The look-up model: <span class="c-loss ea-num">0</span> on the training set and <span class="c-loss ea-num">{e['look_test']}</span> on the test set.""")

    explore_shuffle = L.explore("shuffle", "shuffle", _frame, start=1, low=1, high=8, step=1, role="", task="Go through all eight shuffles: which model does better on the test set, and does that ever change?")
    explore_shuffle
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    In every shuffle the line does better on the test set: the look-up model is far worse there, however perfect it looks on the training set. The test MSE of the line moves with the shuffle, and is sometimes even below its training MSE: 8 houses is a small test set, and which ones end up in it is partly luck.

    ### Recipe

    To find out how a model does on new data:

    1. Shuffle the positions of the examples: `order = rng.permutation(n)`.
    2. Take about 80% of the positions for training and the rest for testing: `train = order[:32]`, `test = order[32:]`.
    3. Select the examples by position: `sizes[train]`, `prices[train]`, and the same with `test`.
    4. Train on the training set only.
    5. At the end, measure the loss once on the test set: that is your estimate for new data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Forty made-up houses: the area in hundreds of m², and a price in thousands of euros of 60 plus 150 per hundred m², up or down by at most 40 by chance. The line is trained with gradient descent, as in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en), with $\eta$ = 0.1. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_split():
    import numpy as np

    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)

    # 1. shuffle the positions
    _order = _rng.permutation(40)
    print(_order[:10])
    # 2. 32 positions for training, 8 for testing
    _train = _order[:32]
    _test = _order[32:]
    # 3. the houses by position
    _x_train = _sizes[_train]
    _y_train = _prices[_train]
    _x_test = _sizes[_test]
    _y_test = _prices[_test]
    print(len(_x_train), len(_x_test))
    # 4. train on the training set only
    _w = 0.0
    _b = 0.0
    for _step in range(2000):
        _r = _y_train - (_w * _x_train + _b)
        _w = _w - 0.1 * (-2 * (_r * _x_train).mean())
        _b = _b - 0.1 * (-2 * _r.mean())
    print(_w, _b)
    # 5. the MSE on the training set, and once on the test set
    print(np.mean((_y_train - (_w * _x_train + _b)) ** 2))
    print(np.mean((_y_test - (_w * _x_test + _b)) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The shuffle puts house 2 first, then house 33, 6 and so on; the first 32 positions go to training. The line learns about 153 per hundred m² and 52 on top, close to the 150 and 60 the houses were made with. Its MSE is about 482 on the training set and about 648 on the test set: on houses it did not learn from, it is off by more here. This is shuffle 1 in the picture.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sorted():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Sorted houses
    """)
    return


@app.cell(hide_code=True)
def choice_sorted_cell():
    choice_sorted = L.choice(
        "sorted",
        question="A file lists 100 houses sorted by area, smallest first. You train on the first 80 rows and test on the last 20, without shuffling. What goes wrong?",
        options={
            "right": "The test set holds only the biggest houses, bigger than every house the model trained on",
            "nothing": "Nothing: 80 to train and 20 to test is a good split",
            "small": "The test set is too small to measure anything",
        },
        correct="right",
        why={
            "nothing": "80 and 20 are fine, but which houses end up in the test set? See *The idea*.",
            "small": "Twenty houses is enough to measure; the trouble is which twenty. See *The idea*.",
            "right": "The test set should look like the new houses the model will meet. Shuffle first, and both sets get small and big houses.",
        },
    )
    choice_sorted
    return (choice_sorted,)


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Split ten houses

        Ten houses, with their areas in `sizes` and their prices in `prices`. Shuffle their positions with `rng` into `order`, put the first 8 positions in `train` and the last 2 in `test`. Replace the three `...`. The last lines select the areas of both sets and show them, so the check sees them. The check also tries other houses.
        """),
        L.task("Split the ten houses into a training set of 8 and a test set of 2."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=11)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "Steps 1 and 2 of the recipe, with 10 houses instead of 40.",
            "`rng.permutation` with a whole number shuffles the positions: which number for ten houses? And how do you take the first 8 of an array, and the rest?",
            "`order = rng.permutation(10)`, `train = order[___]` and `test = order[___]`.",
        ],
        why="With seed 1 the shuffle is `[8 4 7 0 1 2 5 9 6 3]`. The first 8 positions go to training; positions 6 and 3 go to testing: the houses with areas 1.5 and 1.8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two_mse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The MSE twice

        The split from exercise 2: `x_train` and `y_train` hold the 8 training houses, `x_test` and `y_test` the 2 test houses. And a line with $w = 160$ and $b = 50$ that fits the training houses well. Put its mean squared error on the training set in `mse_train`, and on the test set in `mse_test`. Replace the two `...`. The last line shows both, so the check sees them. The check also tries another line.
        """),
        L.task("Put the MSE of the line on the training set in `mse_train` and on the test set in `mse_test`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_two_mse():
    box_two_mse = L.code_box("two-mse", lines=12)
    box_two_mse
    return (box_two_mse,)


@app.cell(hide_code=True)
def check_two_mse(box_two_mse):
    result_two_mse = L.check("two-mse", box_two_mse.value)
    result_two_mse
    return (result_two_mse,)


@app.cell(hide_code=True)
def hints_two_mse():
    L.hints(
        "two-mse",
        [
            "Step 5 of the recipe, for both sets: the mean of the squared residuals.",
            "The prediction for an area `x` is `w * x + b`. Which `x` and `y` belong to each set?",
            "`mse_train = np.mean((y_train - ___) ** 2)`, and `mse_test` the same with `x_test` and `y_test`.",
        ],
        why="On the 8 training houses the residuals lie between −4 and 6, and their squares have a mean of 13.5. On the 2 test houses the line predicts 290 and 338 for 290 and 330: residuals 0 and −8, squares 0 and 64, mean 32.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) the loss of a model was measured on the houses it learned from; now you also measure it on houses it never saw, and only that tells you how it will do on new ones. In [Random numbers and seeds](/course/open/np-random?lang=en) `rng.permutation` shuffled a list; here it shuffles the positions of the examples, and the seed makes the split the same every run. In the next lesson, *Overfitting and underfitting*, you see what happens when a model is too flexible, like the look-up model, or too simple.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sorted, result_split, result_two_mse):
    L.wrap_up(choice_sorted, result_split, result_two_mse)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
