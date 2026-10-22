# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-validation-set. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-validation-set", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_which():
    quick_which = L.choice(
        "quick-which",
        question="`w, b = fit(x, y, lr, steps)` trains a line. Which of these are hyperparameters?",
        options={
            "params": "`w` and `b`",
            "data": "`x` and `y`",
            "right": "`lr` and `steps`",
        },
        correct="right",
        why={
            "params": "Those come out of training: they are parameters. See *The idea* in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).",
            "data": "Those are the training examples, not settings. See *The idea* in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).",
            "right": "You choose `lr` and `steps` before training, and training does not change them.",
        },
    )
    quick_which
    return (quick_which,)


@app.cell(hide_code=True)
def quick_check_how():
    quick_how = L.choice(
        "quick-how",
        question="How do you choose a hyperparameter such as the degree of a curve?",
        options={
            "right": "Train a model with each value, and compare them on examples they did not train on",
            "train": "Take the value with the lowest MSE on the training set",
            "gradient": "Let gradient descent change it, like a weight",
        },
        correct="right",
        why={
            "train": "On the training set more capacity always looks better. See *The idea* in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).",
            "gradient": "Training does not change a hyperparameter. See *The idea* in [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en).",
            "right": "Each value gives its own trained model; examples it did not see show which one works on new data.",
        },
    )
    quick_how
    return (quick_how,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A hyperparameter is chosen on examples the model did not train on. The test set seems an obvious place for that, but the test set has another job: to say, once and honestly, how the finished model does on new data. Use it to choose, and it has helped shape the model; its score then looks better than the model really is. So you set aside a third set of examples, just for choosing.

    **An everyday example.** A school tries three ways of teaching maths and keeps the one whose pupils do best on a mock exam; the real exam stays sealed until the end. If it chose by the real exam, that grade would also reward luck with those questions, and the chosen way would look better than it is.
    """)
    return


@app.cell(hide_code=True)
def predict_best_degree():
    guess_best_degree = L.guess(
        "best-degree",
        question="Nine curves of degree 1 to 9, each trained on the same 12 temperatures, are measured on 10 other temperatures. Which degree do you think gets the lowest MSE there?",
        options={
            "line": "1",
            "two": "2",
            "nine": "9",
        },
        reveal="**Degree 2**, with an MSE of about 0.80 on those 10 temperatures. On the 12 training temperatures, degree 9 would have won, with 0.49, but on the other 10 it gets about 36: it overfits. The rule the temperatures were made with has degree 2.",
    )
    guess_best_degree
    return (guess_best_degree,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Split the examples into three sets, often about 60%, 20% and 20%, after shuffling them:

        - the **training set**, to learn the parameters;
        - the **validation set**, to choose the hyperparameters: train a model with each value on the training set, and keep the one with the lowest loss on the validation set;
        - the **test set**, to measure the chosen model once, at the end.

        The validation set consists of examples the model did not train on, so it shows which value works on new data. The test set stays out of it until everything has been chosen, so its score is an honest estimate.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **validation set** | | examples kept apart from the training set to choose hyperparameters with | `x_val`, `y_val` |
        """),
        L.watch_out("The validation score of the chosen model is also a little too optimistic: among several models, the winner is partly the one that was luckiest on those examples. That is why you still measure on the test set at the end, and report that number."),
        mo.md(r"""
        The temperatures of the worked example below, split into 12 for training, 10 for validation and 10 for testing. For every degree, the <span class="c-loss">MSE</span> on the training set (filled dots) and on the validation set (open dots).
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_degrees():
    _shown = {_d: L.number(_e["val"], 0) for _d, _e in C.ERRORS.items()}
    mo.Html(C.picture({"title": "Training and validation MSE per degree", "degree": "degree", "mse": "MSE", "chosen": "chosen",
                       "alt": "The MSE of degrees 1 to 9. On the training set it falls from 5.2 to 0.49. On the validation set it falls from 8.5 to 0.80 at degree 2, the chosen degree, and then rises to about 36 at degree 9."}, _shown))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To choose a hyperparameter and report an honest score:

    1. Shuffle the examples and split them into a training set, a validation set and a test set.
    2. For each value of the hyperparameter: train on the training set, and measure on the validation set.
    3. Keep the value with the lowest validation loss: `np.argmin` gives its position in the list.
    4. Measure the chosen model once on the test set, and report that number.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The 32 temperatures from [Overfitting and underfitting](/course/open/ml-overfitting?lang=en), now split three ways: the same 12 training temperatures as there, few on purpose so that a flexible curve overfits sooner, and its 20 test temperatures split into 10 to validate and 10 to test. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_validation():
    import numpy as np

    _rng = np.random.default_rng(75)
    _hours = 24 * _rng.random(size=32)
    _temps = 20 - 0.05 * (_hours - 15) ** 2 + 3 * (_rng.random(size=32) - 0.5)

    # 1. shuffle and split: 12 to train, 10 to validate, 10 to test
    _order = _rng.permutation(32)
    _train, _val, _test = _order[:12], _order[12:22], _order[22:]
    # 2. for each degree: train on the training set, measure on the validation set
    _degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    _val_mses = []
    for _degree in _degrees:
        _w = np.polyfit(_hours[_train], _temps[_train], _degree)
        _val_mses.append(np.mean((_temps[_val] - np.polyval(_w, _hours[_val])) ** 2))
    print(np.array(_val_mses))
    # 3. the degree with the lowest validation MSE
    _best = _degrees[np.argmin(_val_mses)]
    print(_best)
    # 4. the chosen model, once on the test set
    _w = np.polyfit(_hours[_train], _temps[_train], _best)
    print(np.mean((_temps[_test] - np.polyval(_w, _hours[_test])) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_train, _val, _test = _order[:12], _order[12:22], _order[22:]` gives each of the three slices its own name, as `x, y = point` did in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en): the commas on the right make a tuple, also without parentheses. The validation MSE falls from about 8.5 at degree 1 to 0.80 at degree 2, and then climbs to about 36 at degree 9. `np.argmin(_val_mses)` is the `.argmin()` from [How wrong is one prediction?](/course/open/ml-residuals?lang=en) written as a function, and it also takes a plain list: it gives position 1, and `_degrees[1]` is degree 2.

    The curve of degree 2 gets about 0.37 on the test set: that is the number you report. That it is lower than its 0.80 on the validation set is chance: with only 10 temperatures per set, an MSE swings a lot from one set to another. The watch-out is about what happens on average.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_test_to_choose():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Choosing on the test set
    """)
    return


@app.cell(hide_code=True)
def choice_test_to_choose_cell():
    choice_test_to_choose = L.choice(
        "test-to-choose",
        question="Someone skips the validation set: they keep the degree with the lowest test MSE, degree 3 with 0.34, and report 0.34. What is wrong with that number?",
        options={
            "nothing": "Nothing: the test set is examples the model did not train on",
            "right": "It is probably too optimistic: the test set helped choose the model",
            "pessimistic": "It is probably too pessimistic: the model did not train on the test set",
        },
        correct="right",
        why={
            "nothing": "It did not train on them, but they did choose the degree. See *Why this matters*.",
            "pessimistic": "Of nine models, the one that did best on these examples was kept. Does that make its score too high or too low? See the watch-out in *The idea*.",
            "right": "Of nine curves, the one that happened to do best on these 10 temperatures was kept, so 0.34 flatters it. There is no honest test score left.",
        },
    )
    choice_test_to_choose
    return (choice_test_to_choose,)


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Choose the degree

        Twelve training measurements in `x_train` (hours) and `y_train` (temperatures), six validation measurements in `x_val` and `y_val`, and the degrees to choose from in `degrees`. In the loop, fit a curve of each degree on the training set and add its validation MSE to `val_mses`. Then put the degree with the lowest validation MSE in `best`. Replace the two `...`. The last line is `best`, so the check sees it. The check also tries other degrees to choose from.
        """),
        L.task("Put the degree with the lowest validation MSE in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose-degree", lines=13)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose-degree", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose-degree",
        [
            "Steps 2 and 3 of the recipe, as in the worked example.",
            "In the loop: `np.polyfit` on the training set, then the MSE on `x_val` and `y_val`. After the loop: `np.argmin` gives a position. What is at that position in `degrees`?",
            "`val_mses.append(np.mean((y_val - np.polyval(w, ___)) ** 2))` and `best = degrees[___]`.",
        ],
        why="The validation MSEs are about 7.87, 0.96, 1.04, 1.01, 1.16 and 2.21. The lowest is at position 1, and `degrees[1]` is 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ### Exercise 3. And then?
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "test-at-end",
        question="You chose degree 2 on the validation set. What do you do with the test set now?",
        options={
            "again": "Compare degree 2 and 3 on it, and keep the better one",
            "right": "Measure the curve of degree 2 on it once, and report that number",
            "train": "Add it to the training set and train again, without measuring",
        },
        correct="right",
        why={
            "again": "Then the test set chooses after all. See step 4 of the recipe.",
            "train": "Then nothing is left to measure the model honestly. See step 4 of the recipe.",
            "right": "The test set has played no part yet, so its score is an honest estimate for new data.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Parameters and hyperparameters](/course/open/ml-hyperparameters?lang=en) you saw that the training loss cannot choose a hyperparameter; the validation set can. In [Overfitting and underfitting](/course/open/ml-overfitting?lang=en) the test MSE showed the pattern; now the test set stays sealed while you choose, so its one score at the end is honest. In the next lesson, *Ask the neighbors: k-nearest neighbors*, you meet a classifier whose hyperparameter is how many neighbors it asks, and you choose it the same way.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_to_choose, result_choose, choice_end):
    L.wrap_up(choice_test_to_choose, result_choose, choice_end)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
