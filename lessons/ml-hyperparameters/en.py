# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-hyperparameters. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-hyperparameters", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en) and [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rate():
    quick_rate = L.choice(
        "quick-rate",
        question="Gradient descent runs with a learning rate that is far too big. What happens to the loss?",
        options={
            "slow": "It falls, but very slowly",
            "right": "It grows, step after step",
            "faster": "It falls faster, to the same lowest point",
        },
        correct="right",
        why={
            "slow": "That is a learning rate that is too small. See the picture in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "faster": "Each step overshoots the lowest point, by more each time. See the picture in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "right": "Each step jumps past the lowest point, further than the step before: the loss grows.",
        },
    )
    quick_rate
    return (quick_rate,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="A curve has an MSE of 0.3 on its training set and 25 on its test set. What does it do?",
        options={
            "under": "It underfits",
            "fine": "It does well",
            "right": "It overfits",
        },
        correct="right",
        why={
            "under": "Then both MSEs would be high. See *The idea* in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "fine": "Then both would be low and close together. See *The idea* in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "right": "Almost exact on the training set, far off on new data: it follows the noise.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A trained model is full of numbers that training has set: its weights and its bias term. But some numbers training never touches: the learning rate, the number of steps, the degree of a curve. You choose those before training starts, and they decide how training goes and what the model can learn. Each kind of number is chosen in its own way, so you need to tell them apart.

    **An everyday example.** A driving instructor decides how many lessons a student gets and how long each one lasts. How hard to brake and when to steer, the student picks up by driving.
    """)
    return


@app.cell(hide_code=True)
def predict_pick_degree():
    guess_pick_degree = L.guess(
        "pick-degree",
        question="Training picks the weights with the lowest MSE on the training set. What do you think happens if you pick the degree of a curve the same way: the degree with the lowest training MSE?",
        options={
            "highest": "It always picks the highest degree you allow",
            "best": "It picks the best degree",
            "lowest": "It always picks degree 1",
        },
        reveal="**It always picks the highest degree you allow.** In [Overfitting and underfitting](/course/open/ml-overfitting?lang=en), more capacity always lowered the training MSE, so this rule picks the most flexible curve: the one most likely to overfit. The degree has to be chosen another way, by how the model does on examples it did not train on.",
    )
    guess_pick_degree
    return (guess_pick_degree,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **<span class="c-param">parameter</span>**, or model parameter, is a number in the model that training adjusts: the weights and the bias term. A **<span class="c-hyper">hyperparameter</span>** is a setting you choose before training, and training does not change it. Some belong to the model itself, such as the degree of a polynomial: it decides how many weights there are. Others steer the training, such as the learning rate $\eta$ and the number of steps. Hyperparameters go into the training code next to the data, parameters come out of it: `w, b = fit(x, y, lr, steps)`.

        Training picks the parameters by lowering the loss on the training set. That cannot work for a hyperparameter such as the degree: on the training set, more capacity always looks better. A hyperparameter is chosen by trying a few values, training a model with each, and comparing those models on examples they did not train on.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **parameter**, model parameter | | a number in a model that training adjusts, such as a weight or the bias term | `w`, `b` |
        | **hyperparameter** | | a setting you choose before training, such as the learning rate or the number of steps; training does not change it | `lr`, `steps` |
        """),
        L.watch_out("Python's documentation also calls the names in a `def` line parameters, such as `lr` and `steps` in `def fit(x, y, lr, steps)`; this course calls them arguments. So `lr` is an argument of `fit` and a hyperparameter of the training, but never a parameter of the model."),
        draw.flow(
            [("You choose", ["hyperparameters:", "learning rate η,", "number of steps"]),
             ("Training", ["gradient descent", "on the training set"]),
             ("Training learns", ["parameters:", "weights w,", "bias term b"])],
            forward=["before\ntraining", "while it\nruns"],
            back=[],
            title="Who sets which numbers"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To tell the parameters and hyperparameters in your own code apart:

    1. Find the numbers you set before training, besides the data, such as `lr` and `steps` in `fit(x, y, lr, steps)`. Those are hyperparameters.
    2. Find the numbers that training gives back, such as `w` and `b`. Those are parameters.
    3. Change a hyperparameter and train again: the parameters come out different.
    4. Training chooses the parameters with the loss on the training set; you choose the hyperparameters by comparing models on examples they did not train on.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The training set of 32 houses from [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en), and gradient descent in a function `_fit` with the hyperparameters `lr` and `steps` as its arguments. The line is trained three times, with 10, 100 and 2000 steps. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_hyperparameters():
    import numpy as np

    def _fit(x, y, lr, steps):
        w = 0.0
        b = 0.0
        for step in range(steps):
            r = y - (w * x + b)
            w = w - lr * (-2 * (r * x).mean())
            b = b - lr * (-2 * r.mean())
        return w, b

    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)
    _order = _rng.permutation(40)
    _x_train = _sizes[_order[:32]]
    _y_train = _prices[_order[:32]]

    # 1. the hyperparameters, chosen before training
    _lr = 0.1
    for _steps in [10, 100, 2000]:
        # 2. and 3. the parameters that come out, for each number of steps
        _w, _b = _fit(_x_train, _y_train, _lr, _steps)
        print(_steps, _w, _b, np.mean((_y_train - (_w * _x_train + _b)) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With 10 steps the line gets $w \approx 128.6$ and $b \approx 85.6$, with a training MSE of about 601. With 100 steps, about 146.1 and 61.6, and 492. With 2000 steps, about 152.8 and 52.3, and 482: the line you got for these houses before. The same training set and the same learning rate, but another number of steps, and the parameters come out different.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Who sets it?

    Put each number in its group: learned by training, or chosen by you.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_who = L.sort("who-sets")
    sort_who
    return (sort_who,)


@app.cell(hide_code=True)
def check_sort(sort_who):
    result_who = L.check("who-sets", sort_who)
    result_who
    return (result_who,)


@app.cell(hide_code=True)
def hints_sort():
    L.hints(
        "who-sets",
        [
            "Steps 1 and 2 of the recipe: for each number, ask whether you set it before training, or training gives it back.",
            "In `w, b = fit(x, y, lr, steps)`, which names go in besides the data, and which come out? And for `w = np.polyfit(x, y, 3)`?",
            "The weight for the area and the bias term come out of `fit`. Which numbers of the polynomial go into `np.polyfit`, and which come out?",
        ],
        why="**Parameters:** the weight for the area, the bias term and the weights of the polynomial; training works them out. **Hyperparameters:** the learning rate, the number of steps and the degree; you choose them before training.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Three learning rates

        Ten houses, and the same function as `_fit` in the worked example, now called `fit`. Train the line once for each learning rate in the list, with `steps` steps, and add its MSE on the houses to `mses`. This training MSE shows how far each run got, not which learning rate to choose. Replace the `...` with two lines inside the loop. The last line is `mses`, so the check sees it. The check also tries another number of steps.
        """),
        L.task("Add the training MSE of the line for each learning rate to `mses`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("three-rates", lines=19)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("three-rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "three-rates",
        [
            "Inside the loop, as in the worked example: train with `fit`, then work out the MSE of the line that comes out.",
            "`fit` gets the sizes, the prices, the learning rate of this round and `steps`, and gives back `w` and `b`. How did you add something to the end of a list?",
            "`w, b = fit(sizes, prices, lr, steps)` and `mses.append(np.mean((prices - (___)) ** 2))`.",
        ],
        why="Each round trains a new line with its own learning rate, and `mses.append` adds its MSE. After 200 steps: about 125 for 0.01, 10.1 for 0.1 and 9.9 for 0.3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_explain():
    mo.md(r"""
    ### Exercise 3. In your own words
    """)
    return


@app.cell(hide_code=True)
def explain_difference_cell():
    explain_difference = L.explain(
        "rate-or-weight",
        question="What is the difference between the learning rate and the weight $w$ of a line? Write one or two sentences.",
        model_answer="The weight w is a parameter: gradient descent changes it at every step, and its value at the end comes from the training set. The learning rate is a hyperparameter: you choose it before training, training does not change it, and it decides how big the steps are that change w.",
        points=[
            "the weight is a parameter, the learning rate a hyperparameter",
            "training changes the weight, step by step, from the training set",
            "you choose the learning rate before training; training does not change it, and it sets how big the steps are",
        ],
    )
    explain_difference
    return (explain_difference,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en) you chose the learning rate yourself and saw what a bad choice does; it is a hyperparameter. In [Overfitting and underfitting](/course/open/ml-overfitting?lang=en) the degree set the capacity of a curve; it is a hyperparameter too, and the training MSE cannot choose it. In the next lesson, *Choosing hyperparameters: the validation set*, you choose them on a separate set of examples, so that the test set stays honest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_who, result_rates, explain_difference):
    L.wrap_up(result_who, result_rates, explain_difference)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
