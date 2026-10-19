# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-overfitting. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-overfitting", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shuffle():
    quick_shuffle = L.choice(
        "quick-shuffle",
        question="Why do you shuffle the examples before you split them into a training set and a test set?",
        options={
            "faster": "So that training goes faster",
            "right": "So that the test set is like the rest of the data, and not, say, only its last part",
            "twice": "So that the model sees every example twice",
        },
        correct="right",
        why={
            "faster": "Shuffling changes which examples go where, not how fast training is. See *The idea* in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).",
            "twice": "After the split, every example is in exactly one of the two sets. See *The idea* in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).",
            "right": "Data often comes sorted; after shuffling, both sets look like the whole.",
        },
    )
    quick_shuffle
    return (quick_shuffle,)


@app.cell(hide_code=True)
def quick_check_test_once():
    quick_test_once = L.choice(
        "quick-test-once",
        question="When do you measure the loss on the test set?",
        options={
            "right": "Once, at the end",
            "every-step": "After every training step",
            "choose": "Whenever you choose between models",
        },
        correct="right",
        why={
            "every-step": "Then the test set would steer the training. See the watch-out in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).",
            "choose": "A test set that helped you choose is no longer an honest estimate. See the watch-out in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).",
            "right": "Once, at the end: then it is an honest estimate for new data.",
        },
    )
    quick_test_once
    return (quick_test_once,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model can miss in two opposite ways. Too simple, and it misses the pattern, like a straight line through temperatures that rise and fall over a day. Too flexible, and it also learns the chance ups and downs of its training examples, which do not come back in new data. The training set and the test set from the previous lesson tell you which of the two you have, and so which way to change the model.

    **An everyday example.** A weather forecaster who always says "12 degrees tomorrow" ignores summer and winter. One who remembers that it was 14.3 degrees at three o'clock on this day last year treats a coincidence as a rule.
    """)
    return


@app.cell(hide_code=True)
def predict_train_mse():
    guess_train_mse = L.guess(
        "train-mse",
        question="You give a curve more and more freedom to bend, step by step, each time trained on the same 12 temperatures. What do you think happens to its MSE on those 12 training temperatures?",
        options={
            "down": "It keeps going down",
            "u": "It goes down, then up again",
            "same": "It stays about the same",
        },
        reveal="**It keeps going down.** In the worked example below, from 5.2 for a straight line to 0.49 for the most flexible curve. A curve with more freedom can always do at least what a stiffer one did, so on the training set it never gets worse. On the test set it goes down and then up again: from 6.8 to 0.6, and then to 21.9.",
    )
    guess_train_mse
    return (guess_train_mse,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **capacity** of a model is how complex the patterns are that it can follow. Here it is set by the **degree** of a curve: linear regression on the features $x, x^2, \dots, x^d$ of one number $x$ gives a curve $w_1 x + w_2 x^2 + \dots + w_d x^d + b$, a **polynomial** of degree $d$. Degree 1 is a straight line; every extra power lets the curve bend once more.

        Measurements also contain **noise**: chance ups and downs that no rule predicts, such as a cloud passing at the moment a temperature is read. Compare the MSE on the training set and on the test set:

        - **Underfitting**: both are high. The model is too simple for the pattern. Give it more capacity.
        - **Overfitting**: the training MSE is low, but the test MSE is much higher. The model follows the noise of its training examples. Give it less capacity, or more training examples.
        - **In between**: both are low and close together. On new data, no model gets below the noise itself.

        `np.polyfit(x, y, d)` finds the weights and the bias term of the polynomial of degree $d$ with the lowest MSE on `x` and `y`, the ones gradient descent would end at after very many steps; it works them out directly, without steps. `np.polyval(w, x)` works out the curve at the numbers in `x`.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **capacity**, model capacity | | how complex the patterns are that a model can follow; more parameters or more flexible features give more capacity | |
        | **polynomial** of degree $d$ | | $w_1 x + w_2 x^2 + \dots + w_d x^d + b$: a sum of powers of $x$, each with its own weight, plus a bias term | `np.polyfit(x, y, d)`, `np.polyval(w, x)` |
        | **noise** | | chance ups and downs in the data that no rule predicts | |
        | **underfitting** | | a model too simple for the pattern: high MSE on the training set and on the test set | |
        | **overfitting** | | a model that follows the noise of its training set: low training MSE, much higher test MSE | |
        """),
        L.watch_out("A training MSE close to 0 is not good news by itself: with about as many numbers to learn as training examples, a curve can pass through almost every point. Always look at the test MSE as well."),
        mo.md(r"""
        One chart shows the temperatures of the worked example below, with a <span class="c-pred">curve</span> of degree $d$: filled dots are the 12 training measurements, open dots the 20 test measurements. The other shows the <span class="c-loss">MSE</span> of every degree: filled dots on the training set, open dots on the test set, larger at degree $d$. Move the slider, or drag across the MSE chart, to change $d$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_degree_cell():
    def _frame(d):
        d = int(d)
        train, test = C.ERRORS[d]
        return (C.picture(d, {"data": f"Degree {d}", "hour": "hour of the day", "temp": "temperature (°C)",
                              "mse": "MSE per degree", "degree": "degree", "loss": "MSE",
                              "alt": f"32 temperatures over a day, rising to about 20 °C in the afternoon, with a curve of degree {d}. The MSE of degrees 1 to 9, the dots of degree {d} larger: on the training set it falls from 5.2 to 0.49; on the test set it falls to about 0.6 at degree 2, stays close to that up to degree 5, and rises to 21.9 at degree 9."}),
                f"""Degree {d}: an MSE of <span class="c-loss ea-num">{L.number(train, 2)}</span> on the training set and <span class="c-loss ea-num">{L.number(test, 2)}</span> on the test set.""")

    explore_degree = L.explore("degree", "d", _frame, start=1, low=1, high=9, step=1, role="", task="Find the degree with the lowest MSE on the test set.")
    explore_degree
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The MSE on the test set is lowest at degree 2, and from degree 6 on it climbs fast. The MSE on the training set keeps going down, all the way to degree 9: a lower MSE on the training set does not mean a better model.

    ### Recipe

    To find out whether a model underfits or overfits:

    1. Train it on the training set.
    2. Measure its MSE on the training set and on the test set.
    3. Both high: underfitting. Give the model more capacity, such as a higher degree or more features.
    4. Training MSE low, test MSE much higher: overfitting. Give it less capacity, or more training examples.
    5. Once you change the model because of the test MSE, the test set has helped you choose, and its MSE is no longer an honest estimate for new data. A later lesson keeps a third set apart for that choice; here the test set only shows the pattern.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Thirty-two made-up temperatures, read at random hours $h$ of one day: they follow the rule $20 - 0.05\,(h - 15)^2$, highest at three in the afternoon, plus noise of at most 1.5 °C up or down. Twelve of them are the training set, the other twenty the test set: few training examples on purpose, so that a flexible curve overfits sooner. Then three curves: degree 1, 2 and 9. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_overfitting():
    import numpy as np

    _rng = np.random.default_rng(75)
    _hours = 24 * _rng.random(size=32)
    _temps = 20 - 0.05 * (_hours - 15) ** 2 + 3 * (_rng.random(size=32) - 0.5)
    _order = _rng.permutation(32)
    _x_train = _hours[_order[:12]]
    _y_train = _temps[_order[:12]]
    _x_test = _hours[_order[12:]]
    _y_test = _temps[_order[12:]]

    for _degree in [1, 2, 9]:
        # 1. train a curve of this degree on the training set
        _w = np.polyfit(_x_train, _y_train, _degree)
        # 2. its MSE on the training set and on the test set
        _mse_train = np.mean((_y_train - np.polyval(_w, _x_train)) ** 2)
        _mse_test = np.mean((_y_test - np.polyval(_w, _x_test)) ** 2)
        print(_degree, _mse_train, _mse_test)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Degree 1, a straight line: about 5.2 on the training set and 6.8 on the test set, both high. It underfits: a line cannot rise and fall. Degree 2: about 0.76 and 0.58, both low and close together, near the noise itself; the rule the temperatures were made with has degree 2 too. Degree 9: only 0.49 on the training set, but about 21.9 on the test set. It overfits: it bends through the chance ups and downs of its 12 training measurements, and between and beyond them it swings far away.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which one overfits?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-overfits",
        question="Three models predict house prices. Their MSE on the training set and on the test set: A has 900 and 950, B has 400 and 450, C has 20 and 1300. Which one overfits?",
        options={
            "a": "A",
            "b": "B",
            "right": "C",
        },
        correct="right",
        why={
            "a": "Both its MSEs are high and close together. Which of the two does that point to? See *The idea*.",
            "b": "Both its MSEs are low and close together. See *The idea*.",
            "right": "C is almost exact on its training houses, but far off on new ones: it follows the noise. A is high on both: it underfits.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_test_mse():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The test MSE of a curve

        Twelve training measurements and six test measurements, and a degree in `degree`. Fit a curve of that degree on the training set with `np.polyfit`, and put its MSE on the test set in `mse_test`. Replace the two `...`. The last line is `mse_test`, so the check sees it. The check also tries another degree. Want to see degree 9? Copy your code into the scratchpad and change `degree` there.
        """),
        L.task("Put the test MSE of a curve of degree `degree`, fitted on the training set, in `mse_test`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_test_mse():
    box_test_mse = L.code_box("test-mse", lines=11)
    box_test_mse
    return (box_test_mse,)


@app.cell(hide_code=True)
def check_test_mse(box_test_mse):
    result_test_mse = L.check("test-mse", box_test_mse.value)
    result_test_mse
    return (result_test_mse,)


@app.cell(hide_code=True)
def hints_test_mse():
    L.hints(
        "test-mse",
        [
            "Steps 1 and 2 of the recipe, but only the test MSE.",
            "`np.polyfit` gets the training hours, the training temperatures and the degree. Which hours go into `np.polyval` to predict the test temperatures?",
            "`w = np.polyfit(x_train, y_train, degree)` and `mse_test = np.mean((y_test - np.polyval(w, ___)) ** 2)`.",
        ],
        why="`np.polyfit` learns from the 12 training measurements only. `np.polyval(w, x_test)` predicts the 6 test temperatures, and the mean of the squared residuals is about 0.96.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 3. Why it swings
    """)
    return


@app.cell(hide_code=True)
def explain_why_cell():
    explain_why = L.explain(
        "why-swings",
        question="In the worked example, the curve of degree 9 has a lower MSE than the curve of degree 2 on the training set, but a much higher one on the test set. Why? Write one or two sentences.",
        model_answer="With nine weights and a bias term, ten numbers to learn, and only twelve training measurements, the curve of degree 9 can bend through almost every training point, so it also follows their noise. Between and beyond those points it swings away from the real rise and fall of the day, so on the test hours it is far off.",
        points=[
            "the curve of degree 9 can bend through almost every training point",
            "so it also follows the noise of the training measurements",
            "between and beyond those points it swings away from the real pattern, so it is far off on the test set",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en) the test set showed that a look-up model, never off on its training houses, does badly on new ones; that was overfitting. Now you can turn the capacity of a model up and down, and see both ways to miss: underfitting when it is too simple, overfitting when it is too flexible. In the next lesson, *Parameters and hyperparameters*, you meet the settings you choose yourself, such as the degree here or the learning rate.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_test_mse, explain_why):
    L.wrap_up(choice_which, result_test_mse, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
