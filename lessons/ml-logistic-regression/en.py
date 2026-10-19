# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-logistic-regression. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-logistic-regression", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) and [Many features: the dot product](/course/open/math-dot-product?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logit():
    quick_logit = L.choice(
        "quick-logit",
        question="A logit of 3 goes through the sigmoid. Roughly what probability comes out?",
        options={
            "low": "0.05",
            "right": "0.95",
            "same": "3",
        },
        correct="right",
        why={
            "low": "That is $\\sigma(-3)$. Is 3 above or below 0? See the picture in [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "same": "The sigmoid turns every number into one between 0 and 1. See *The idea* in [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "right": "$e^{-3}$ is about 0.05, so $\\sigma(3) = 1 / 1.05 \\approx 0.95$: a positive logit gives more than 0.5.",
        },
    )
    quick_logit
    return (quick_logit,)


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="What does `np.dot([2, 5], [3, 4])` give?",
        options={
            "products": "`array([ 6, 20])`",
            "sum": "`14`",
            "right": "`26`",
        },
        correct="right",
        why={
            "products": "Those are the products. What is the last step of a dot product? See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "sum": "That adds up all four numbers. What happens to the numbers in the same place first? See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "right": "2 × 3 + 5 × 4 = 6 + 20 = 26.",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) the logit was given. A model has to work it out itself, from the features of each example. You already know a way: in [Many features: the dot product](/course/open/math-dot-product?lang=en), a model gave each feature its own weight and added it all up. Put the sigmoid on top, and you have logistic regression: a model for two classes that gives a probability. It is one of the most used models there is, and later in the course it is the last step of many neural networks.

    **An everyday example.** A teacher guesses who will pass an exam. Every hour of study counts, sleep counts a bit too, and there is a bar to clear. A student far above the bar will very likely pass; for one just above it, it could still go either way.
    """)
    return


@app.cell(hide_code=True)
def predict_student():
    guess_student = L.guess(
        "student",
        question="A model for the students of [Examples, features and labels](/course/open/ml-vocabulary?lang=en) has the weights 1 for each hour of study and 0.5 for each hour of sleep, and the bias term −8. A student studied 6 hours and slept 6. What do you think the model predicts?",
        options={
            "pass": "A pass",
            "fail": "A fail",
            "half": "Exactly 50/50",
        },
        reveal="**A pass, with about 0.73.** The logit is 1 × 6 + 0.5 × 6 − 8 = 1: above 0, so the probability is above 0.5. $\\sigma(1)$ is about 0.73.",
    )
    guess_student
    return (guess_student,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Logistic regression** works out the logit as the dot product of the weights and the features, plus the bias term, and puts it through the sigmoid. The <span class="c-pred">prediction</span> $\hat{y}$ is now a probability: the probability of class 1, such as a pass.
        """),
        L.formula(r"[[pred:\hat{y}]] = \sigma([[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]])"),
        mo.md(r"""
        The model predicts class 1 where $\hat{y}$ is above 0.5. That is exactly where the logit is above 0. Where the logit is exactly 0, the model is unsure: 0.5. Those points form the **decision boundary**: on one side the model predicts class 1, on the other class 0.

        With two features, $x_1$ and $x_2$, the decision boundary is the straight line $w_1 x_1 + w_2 x_2 + b = 0$. The weights tilt the line; the bias term moves it.

        To find the point on the line for a given $x_2$, solve for $x_1$. Move the other two terms to the other side of the $=$, which flips their signs: $w_1 x_1 = -b - w_2 x_2$. Then divide by $w_1$: $x_1 = (-b - w_2 x_2) / w_1$. With the weights 1 and 0.5 and $b = -8$, a student who slept 6 hours is on the line at $x_1 = (8 - 0.5 \times 6) / 1 = 5$ hours of study: 5 + 3 − 8 = 0.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **logistic regression** | | a model for two classes: the sigmoid of the dot product of weights and features, plus the bias term | `1 / (1 + np.exp(-(np.dot(w, x) + b)))` |
        | **decision boundary** | | where the logit is 0 and the probability 0.5: the line between the two classes the model predicts | `np.dot(w, x) + b == 0` |
        """),
        L.watch_out("Despite its name, logistic regression does classification: it predicts a class, with a probability. The word *regression* in its name comes from the weighted sum inside, which is the same as in a model that predicts a number."),
        mo.md(r"""
        The students of [Examples, features and labels](/course/open/ml-vocabulary?lang=en) whose result is known, by hours of study and of sleep. The filled dots passed, the open dots failed. The <span class="c-pred">blue line</span> is the decision boundary of the model with the weights 1 and 0.5. Move the slider: it sets the <span class="c-param">bias term</span> $b$, and moves the line. The other chart counts, for every $b$, the students on the <span class="c-loss">wrong side</span> of the line (filled) and exactly on it (open); drag across it to set $b$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        wrong, on = C.sides(b)
        on_text = f", and {on} exactly on it, where the model is unsure" if on else ""
        return (C.picture(b, {"title": "Students and the decision boundary", "hours": "hours of study", "sleep": "hours of sleep",
                              "above": "pass", "below": "fail", "wrong": "On the wrong side, for every b", "count": "students", "on": "on the line",
                              "alt": f"Seven students by hours of study and sleep: four filled dots that passed, three open dots that failed, and the decision boundary as a straight line that falls from upper left to lower right. With b = {L.number(b, 1)}, the model puts {wrong} of the 7 students on the wrong side of the line{on_text}. A second chart shows, for every b on the slider, how many students lie on the wrong side (filled dots) and exactly on the line (open dots), with a vertical line at the slider's b: 2 on the wrong side at −10 and −9.5; 2 on the line at −9; from −8.5 to −7.5 none on the wrong side and none on the line; 1 on the line at −7; and 1 on the wrong side at −6.5 and −6."}),
                f"""With <i>b</i> = {L.number(b, 1)}, the model puts <span class="ea-num">{wrong}</span> of the 7 students on the wrong side of the line{on_text}.""")

    explore_b = L.explore("boundary-b", "b", _frame, start=-6, low=-10, high=-6, step=0.5, role="param", task="Find the values of $b$ that put every student on the correct side of the line.")
    explore_b
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With $b$ between −8.5 and −7.5, every student is on the correct side. At −9 and −7, no student is on the wrong side, but some lie exactly on the line, where the model is unsure. Training, in a later lesson, finds such weights and such a bias term by itself.

    ### Recipe

    To let logistic regression predict a class:

    1. Put the features of the example in a vector $\mathbf{x}$, and take the weights $\mathbf{w}$ and the bias term $b$ of the model.
    2. Work out the logit: `z = np.dot(w, x) + b`.
    3. Put it through the sigmoid: `1 / (1 + np.exp(-z))`.
    4. Above 0.5, the model predicts class 1: the example lies on that side of the decision boundary.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The model with the weights 1 and 0.5 and the bias term −8, first for one student who studied 5 hours and slept 8, then for the three students of [Examples, features and labels](/course/open/ml-vocabulary?lang=en) whose result is not known yet. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_logistic():
    import numpy as np

    # 1. the model, and one student: 5 hours of study, 8 of sleep
    _w = np.array([1, 0.5])
    _b = -8
    _x = np.array([5, 8])
    # 2. the logit
    _z = np.dot(_w, _x) + _b
    print(_z)
    # 3. the probability of a pass
    _p = 1 / (1 + np.exp(-_z))
    print(_p)
    # 4. the class
    print(_p > 0.5)
    # 2. to 4. three students at once: hours and sleep each in an array
    _hours = np.array([4, 7, 2])
    _sleep = np.array([7, 6, 9])
    _z3 = _w[0] * _hours + _w[1] * _sleep + _b
    _p3 = 1 / (1 + np.exp(-_z3))
    print(_p3)
    print(_p3 > 0.5)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The student with 5 and 8 gets the logit 5 + 4 − 8 = 1 and a probability of 0.73: a pass, and that is what happened. For the three new students, the logits are −0.5, 2 and −1.5: the model gives 0.38, 0.88 and 0.18, so it predicts a pass only for the student with 7 hours of study. For many students at once, each feature is an array, and `_w[0] * _hours + _w[1] * _sleep` works out every dot product in one line.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_boundary():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. On the line
    """)
    return


@app.cell(hide_code=True)
def choice_boundary_cell():
    choice_boundary = L.choice(
        "on-boundary",
        question="The model has the weights 1 for the hours of study and 0.5 for the hours of sleep, and the bias term −8. Which student lies exactly on the decision boundary?",
        options={
            "five-eight": "5 hours of study, 8 of sleep",
            "eight-four": "8 hours of study, 4 of sleep",
            "right": "4 hours of study, 8 of sleep",
            "two-six": "2 hours of study, 6 of sleep",
        },
        correct="right",
        why={
            "five-eight": "That is the student of the worked example: a logit of 1. What is the logit on the decision boundary? See *The idea*.",
            "eight-four": "Work out the logit: 8 + 2 − 8 = 2. What is the logit on the decision boundary? See *The idea*.",
            "two-six": "Work out the logit: 2 + 3 − 8 = −3. Which side of the line is that? See *The idea*.",
            "right": "1 × 4 + 0.5 × 8 − 8 = 0: the logit is 0, so the probability is exactly 0.5.",
        },
    )
    choice_boundary
    return (choice_boundary,)


@app.cell(hide_code=True)
def text_exercise_probability():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The probability of one example

        Finish `probability(x, w, b)`: it gets the features `x` of one example, the weights `w` and the bias term `b`, and gives back the probability that logistic regression gives. Replace the `...`. The check calls `probability([7, 6], [1, 0.5], -8)`: that should give about `0.881`. It also tries other weights, and three features.
        """),
        L.task("Make `probability` give back the probability of class 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probability():
    box_probability = L.code_box("probability", lines=7)
    box_probability
    return (box_probability,)


@app.cell(hide_code=True)
def check_probability(box_probability):
    result_probability = L.check("probability", box_probability.value)
    result_probability
    return (result_probability,)


@app.cell(hide_code=True)
def hints_probability():
    L.hints(
        "probability",
        [
            "Steps 2 and 3 of the recipe: first the logit, then the sigmoid.",
            "The logit is the dot product of `w` and `x`, with something added. What is added, and where does it go: inside or outside the sigmoid?",
            "`z = np.dot(w, x) + ___` and `return 1 / (1 + np.exp(___))`.",
        ],
        why="`np.dot(w, x) + b` is the logit, and `1 / (1 + np.exp(-z))` its sigmoid. For `[7, 6]`: 7 + 3 − 8 = 2, and $\\sigma(2)$ is about 0.881.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How much study is enough?

        A student slept 7 hours. How many hours of study put them exactly on the decision boundary of the model? With more, the model predicts a pass. Put it in `hours`. Replace the `...`. The last line is `hours`, so the check sees it. The check also tries other weights.
        """),
        L.task("Put the hours of study on the decision boundary for `sleep` hours of sleep in `hours`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("hours-needed", lines=8)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("hours-needed", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "hours-needed",
        [
            "On the decision boundary, the logit is 0: `w[0] * hours + w[1] * sleep + b = 0`.",
            "Solve that for `hours`: move the other two terms to the other side of `= 0`. What happens to their signs? Then what do you divide by?",
            "`hours = (-b - ___) / ___`.",
        ],
        why="From `w[0] * hours + w[1] * sleep + b = 0` follows `hours = (-b - w[1] * sleep) / w[0]`: (8 − 3.5) / 1 = 4.5.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Many features: the dot product](/course/open/math-dot-product?lang=en), the dot product plus the bias term gave a prediction; in logistic regression it gives the logit. In [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) the sigmoid turned a logit into a probability; now the logit comes from the features. In this lesson the weights were given; training finds them, in a later lesson. In the next lesson, *Scoring probabilities with the log: log loss*, you measure how good a model's probabilities are, the first step towards training it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_boundary, result_hours, result_probability):
    L.wrap_up(choice_boundary, result_probability, result_hours)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
