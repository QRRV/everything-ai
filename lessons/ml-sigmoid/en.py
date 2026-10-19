# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-sigmoid. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow logistic function

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-sigmoid", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Exponents and logarithms](/course/open/math-exp-log?lang=en), [Probability: counting how often](/course/open/prob-basics?lang=en) and [Classification: predicting a class](/course/open/ml-classification?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_exp():
    quick_exp = L.choice(
        "quick-exp",
        question="How big is $e^{-z}$ when $z$ is large, such as 10?",
        options={
            "negative": "Negative",
            "right": "Close to 0, but above 0",
            "large": "Very large",
        },
        correct="right",
        why={
            "negative": "Is a power of $e$ ever negative? See *The idea* in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "large": "That is $e^{z}$. What does a negative exponent do? See *The idea* in [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "right": "$e^{-10} = 1/e^{10}$, about 0.000045: tiny, but never 0 and never negative.",
        },
    )
    quick_exp
    return (quick_exp,)


@app.cell(hide_code=True)
def quick_check_probability():
    quick_probability = L.choice(
        "quick-probability",
        question="Which of these numbers can be a probability?",
        options={
            "above": "1.3",
            "below": "−0.2",
            "right": "0.7",
        },
        correct="right",
        why={
            "above": "A probability is a share of all tries. Can a share be more than all of them? See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "below": "A probability counts how often something happens. Can that be less than never? See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "right": "A probability lies between 0, never, and 1, always.",
        },
    )
    quick_probability
    return (quick_probability,)


@app.cell(hide_code=True)
def quick_check_task():
    quick_task = L.choice(
        "quick-task",
        question="A model predicts whether a mail is spam or not. Which kind of task is that?",
        options={
            "right": "Classification, with two classes",
            "regression": "Regression",
        },
        correct="right",
        why={
            "regression": "Is the label a number, or one of a fixed set of answers? See the recipe in [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "right": "The label is one of two answers, spam or not spam: two classes.",
        },
    )
    quick_task
    return (quick_task,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A spam filter should say how sure it is: a probability between 0 and 1. But a model for classification first works out a score for each mail, and a score can be any number: −7, 0.3 or 12. The higher it is, the more the mail looks like spam. Later lessons show how a model gets its score; this lesson is about the last step: how do you turn any number into a probability? You need a function that turns every number into one between 0 and 1, that gives a higher probability to a higher score, and that gives 0.5 to a score of 0, which leans neither way.

    **An everyday example.** A quiz asks a yes-or-no question. Without a clue, it is 50/50. Each clue for yes makes you surer, and each clue against makes you less sure, but no clue makes you completely sure: you get closer and closer to 100% or to 0%, and never reach it.
    """)
    return


@app.cell(hide_code=True)
def predict_zero():
    guess_zero = L.guess(
        "zero",
        question="The function of this lesson is $\\sigma(z) = \\dfrac{1}{1 + e^{-z}}$. What do you think $\\sigma(0)$ is?",
        options={
            "zero": "0",
            "half": "0.5",
            "one": "1",
        },
        reveal="**0.5.** $e^{0} = 1$, so $\\sigma(0) = 1 / (1 + 1) = 0.5$. A score of 0 leans neither way: the probability is as large as that of the opposite.",
    )
    guess_zero
    return (guess_zero,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **sigmoid** turns any number $z$ into a number between 0 and 1:
        """),
        L.formula(r"[[pred:\sigma(z)]] = \frac{1}{1 + e^{-z}}"),
        mo.md(r"""
        For a large $z$, $e^{-z}$ is close to 0, and $\sigma(z)$ close to 1. For a very negative $z$, $e^{-z}$ is huge, and $\sigma(z)$ close to 0. At $z = 0$ it is exactly 0.5. A higher $z$ always gives a higher $\sigma(z)$, and it never reaches 0 or 1, though a computer rounds it to exactly 1 from about $z = 37$. It is also symmetric: $\sigma(-z) = 1 - \sigma(z)$. Statistics calls it the logistic function.

        The score $z$ that goes in is called the **logit**. For a spam filter, $\sigma(z)$ is the probability that the mail is spam, and $1 - \sigma(z)$ that it is not. Above 0.5, the filter leans to spam: that happens exactly when the logit is above 0.

        The way back works too: the logit of a probability $p$ is $z = \log\left(\frac{p}{1 - p}\right)$. $\frac{p}{1 - p}$ is called the odds: how many times as likely the class is as not the class.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **sigmoid**, $\sigma(z)$ | sigma of z, the sigmoid | $1 / (1 + e^{-z})$: turns any number into one between 0 and 1 | `1 / (1 + np.exp(-z))` |
        | **logit**, $z$ | | the score that goes into the sigmoid: any number, positive or negative | `z` |
        """),
        L.watch_out("Logit means two things. In this course, it is the score before the sigmoid. Statistics uses it for the way back, $\\log(p / (1 - p))$, also called the log-odds. For two classes they give the same number: the log-odds of $\\sigma(z)$ is $z$ again."),
        mo.md(r"""
        Move the slider, or drag across the chart, to set the logit $z$. The point shows the <span class="c-pred">probability</span> $\sigma(z)$ that the sigmoid gives it.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        e = math.exp(-z)
        p = C.sigmoid(z)
        return (C.picture(z, {"title": "The sigmoid", "z": "logit z", "p": "probability",
                              "alt": f"The sigmoid, an S-shaped curve from near 0 on the left to near 1 on the right, through 0.5 at z = 0. At z = {L.number(z, 1)} there is a point at {L.number(p, 3)}."}),
                f"""At <i>z</i> = {L.number(z, 1)}: <i>e</i><sup>−<i>z</i></sup> = {L.number(e, 4)}, so <i>σ</i>(<i>z</i>) = 1 / (1 + {L.number(e, 4)}) = <span class="c-pred ea-num">{L.number(p, 3)}</span>.""")

    explore_z = L.explore("logit-z", "z", _frame, start=1, low=-6, high=6, step=0.5, role="", task="From which $z$ on is $\\sigma(z)$ above 0.99, and up to which $z$ is it below 0.01?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    From $z = 5$ on, $\sigma(z)$ is above 0.99, and up to $z = -5$, below 0.01. There, far from $z = 0$, the curve is almost flat: a step of 0.5 in $z$ changes the probability by less than 0.005. Around $z = 0$ it is steepest: one step there changes it by 0.12.

    ### Recipe

    To turn a score into a probability:

    1. Take the logit $z$: any number.
    2. Work out $e^{-z}$: `np.exp(-z)`.
    3. Add 1, and divide 1 by it: $\sigma(z)$ = `1 / (1 + np.exp(-z))`.
    4. Read it as the probability of the class: above 0.5 the model leans to it, below 0.5 away from it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Five mails with the logits −3, −1, 0, 1 and 3, all at once with NumPy. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_sigmoid():
    import numpy as np

    # 1. the logits of five mails
    _z = np.array([-3, -1, 0, 1, 3])
    # 2. e to the power -z
    _e = np.exp(-_z)
    print(_e)
    # 3. 1 divided by 1 plus that
    _p = 1 / (1 + _e)
    print(_p)
    # 4. which mails the filter leans to call spam
    print(_p > 0.5)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $e^{-z}$ goes from 20.1 for $z = -3$ down to 0.05 for $z = 3$. The probabilities go from 0.047 to 0.953, with 0.5 in the middle. The mails with logits −1 and 1 get 0.269 and 0.731: together 1, because the sigmoid is symmetric. Only the mails with a logit above 0 get a probability above 0.5; the mail with logit 0 sits exactly at 0.5, so it is not above it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minus_two():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A logit of −2
    """)
    return


@app.cell(hide_code=True)
def choice_minus_two_cell():
    choice_minus_two = L.choice(
        "minus-two",
        question="A spam filter gives a mail the logit −2. Which probability of spam does that give?",
        options={
            "plus": "About 0.88",
            "right": "About 0.12",
            "negative": "−2: a logit is already a probability",
            "half": "0.5",
        },
        correct="right",
        why={
            "plus": "That is $\\sigma(2)$. Is the logit here above or below 0? See the picture.",
            "negative": "A probability lies between 0 and 1, and a logit can be any number. See the table of new words.",
            "half": "0.5 belongs to a logit of 0. What does a negative logit do? See the picture.",
            "right": "$e^{2}$ is about 7.39, so $\\sigma(-2) = 1 / (1 + 7.39) \\approx 0.12$. So the filter leans towards not spam.",
        },
    )
    choice_minus_two
    return (choice_minus_two,)


@app.cell(hide_code=True)
def text_exercise_sigmoid():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The sigmoid as a function

        Finish `sigmoid(z)`: it gets a logit and gives back its probability. Replace the `...`. The check calls `sigmoid(0)`: that should give `0.5`. It also tries other logits, positive and negative.
        """),
        L.task("Make `sigmoid` give back the sigmoid of `z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sigmoid():
    box_sigmoid = L.code_box("sigmoid", lines=6)
    box_sigmoid
    return (box_sigmoid,)


@app.cell(hide_code=True)
def check_sigmoid(box_sigmoid):
    result_sigmoid = L.check("sigmoid", box_sigmoid.value)
    result_sigmoid
    return (result_sigmoid,)


@app.cell(hide_code=True)
def hints_sigmoid():
    L.hints(
        "sigmoid",
        [
            "The formula in *The idea*, in Python: steps 2 and 3 of the recipe.",
            "Which NumPy function gives $e$ to a power? Mind the minus sign, and which part is the bottom of the fraction.",
            "`return 1 / (1 + ___)`.",
        ],
        why="`np.exp(-z)` is $e^{-z}$, and `1 / (1 + np.exp(-z))` divides 1 by 1 plus that. For `z = 0`: 1 / (1 + 1) = 0.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_logit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Back to the logit

        A filter gives a mail a probability of spam of 0.9. Which logit did it have? Put it in `z`, with the way back from *The idea*. Replace the `...`. The last line is `z`, so the check sees it. The check also tries another probability.
        """),
        L.task("Put the logit of `p` in `z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logit():
    box_logit = L.code_box("back-to-logit", lines=6)
    box_logit
    return (box_logit,)


@app.cell(hide_code=True)
def check_logit(box_logit):
    result_logit = L.check("back-to-logit", box_logit.value)
    result_logit
    return (result_logit,)


@app.cell(hide_code=True)
def hints_logit():
    L.hints(
        "back-to-logit",
        [
            "First the odds: how many times as likely spam is as not spam. Then the log of that.",
            "The probability of not spam is `1 - p`. Which NumPy function gives the log with base $e$?",
            "`z = np.log(p / ___)`.",
        ],
        why="The odds are 0.9 / 0.1 = 9, and the log of 9 is about 2.2. Check it: $\\sigma(2.2)$ is about 0.9.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Exponents and logarithms](/course/open/math-exp-log?lang=en), $e^{x}$ was always positive and grew faster and faster; the sigmoid uses that to squeeze every number between 0 and 1, and the log takes you back. In [Probability: counting how often](/course/open/prob-basics?lang=en) a probability lay between 0 and 1; now a model can give one. In [Classification: predicting a class](/course/open/ml-classification?lang=en) a label was one of a fixed set of classes; with two classes, the sigmoid gives the probability of one of them. In the next lesson, *Logistic regression and the decision boundary*, a model works out the logit from the features, and the sigmoid turns it into a probability.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_minus_two, result_logit, result_sigmoid):
    L.wrap_up(choice_minus_two, result_sigmoid, result_logit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
