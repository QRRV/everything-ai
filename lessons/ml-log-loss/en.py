# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-log-loss. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow surprisal, binary cross-entropy

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-log-loss", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) and [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_boundary():
    quick_boundary = L.choice(
        "quick-boundary",
        question="A logistic regression has the weights 2 and 1 and the bias term −4. Where does the example with the features 1 and 2 lie?",
        options={
            "right": "Exactly on the decision boundary",
            "pass": "On the side of class 1",
            "fail": "On the side of class 0",
        },
        correct="right",
        why={
            "pass": "Work out the logit: 2 × 1 + 1 × 2 − 4. Which side has a logit above 0? See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "fail": "Work out the logit: 2 × 1 + 1 × 2 − 4. Which side has a logit below 0? See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "right": "The logit is 2 + 2 − 4 = 0, so the probability is exactly 0.5: on the decision boundary.",
        },
    )
    quick_boundary
    return (quick_boundary,)


@app.cell(hide_code=True)
def quick_check_mse():
    quick_mse = L.choice(
        "quick-mse",
        question="Two examples have the labels 4 and 6, and a model predicts 3 and 4. What is the mean squared error?",
        options={
            "sum": "5",
            "plain": "1.5",
            "right": "2.5",
        },
        correct="right",
        why={
            "sum": "That is the sum of the squared residuals. What does the *mean* in the name ask for? See the formula in [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).",
            "plain": "That is the mean of the residuals, without squaring them. What does *squared* in the name ask for? See the formula in [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en).",
            "right": "The residuals are 1 and 2; squared, 1 and 4; their mean is 2.5.",
        },
    )
    quick_mse
    return (quick_mse,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To train a model, you need a loss: one number for how wrong it is. For a number, the mean squared error worked well. For a probability, it does not. A spam filter that says 0.01 for a mail that is spam is confidently wrong, yet its squared residual, $(1 - 0.01)^2 = 0.98$, is not even four times the 0.25 of a filter that said 0.5 and did not know, and no probability can make it pass 1. A good loss for probabilities punishes a confident mistake much harder than an unsure one.

    **An everyday example.** A weather forecaster who says 99% sun on a day it pours loses far more trust than one who said 60% sun. Both were wrong, but one was sure of it.
    """)
    return


@app.cell(hide_code=True)
def predict_surprise():
    guess_surprise = L.guess(
        "surprise",
        question="A model gave probability 0.5 to what really happened, and that costs $-\\log(0.5) \\approx 0.69$. What do you think it costs when the model gave 0.01 to what really happened, with the same formula?",
        options={
            "one": "About 1",
            "five": "About 4.6",
            "infinite": "Infinitely much",
        },
        reveal="**About 4.6.** $-\\log(0.01) = \\log(100) \\approx 4.6$: almost seven times as much as for 0.5. Only a probability of exactly 0 would cost infinitely much.",
    )
    guess_surprise
    return (guess_surprise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Look at the probability the model gave to what really happened, and take minus its log. A probability of 1 costs 0; the smaller the probability, the more it costs, without limit. $-\log(p)$ is called the **surprise** of an outcome with probability $p$: a likely outcome surprises little, an unlikely one a lot. Information theory calls it the surprisal.

        For an example with the label $y$, 1 for spam and 0 for not spam, and the <span class="c-pred">predicted probability</span> $\hat{y}$ of spam, the model gave $\hat{y}$ to spam and $1 - \hat{y}$ to not spam. One formula covers both, because one of its two terms is always 0. The **log loss** is the surprise of what really happened, averaged over all $n$ examples:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\frac{1}{n} \sum_{i=1}^{n} \left( y_i \log [[pred:\hat{y}_i]] + (1 - y_i) \log (1 - [[pred:\hat{y}_i]]) \right)"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **surprise** | | $-\log(p)$: how surprising an outcome with probability $p$ is | `-np.log(p)` |
        | **log loss** | | the mean surprise of what really happened: a loss for predicted probabilities of two classes | `loss` |
        """),
        L.watch_out("The log loss needs the log of the probability of what really happened: $\\hat{y}$ for a spam mail, but $1 - \\hat{y}$ for a mail that is not spam. PyTorch calls this loss binary cross-entropy."),
        mo.md(r"""
        Move the slider, or drag across the chart, to set the <span class="c-pred">probability of spam</span> $\hat{y}$ that the model gives a mail that is spam. The solid line is the <span class="c-loss">log loss</span>, the dashed line the <span class="c-loss">squared residual</span> of the mean squared error.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_p_cell():
    def _frame(p):
        log = -math.log(p)
        sq = (1 - p) ** 2
        return (C.picture(p, {"title": "The loss of one spam mail", "p": "probability of spam ŷ", "loss": "loss",
                              "log": "log loss", "squared": "squared residual",
                              "alt": f"For a spam mail: the log loss shoots up as the probability of spam goes to 0, while the squared residual stays below 1. At ŷ = {L.number(p, 2)} the log loss is {L.number(log, 2)} and the squared residual {L.number(sq, 2)}."}),
                f"""At <i>ŷ</i> = {L.number(p, 2)}: the log loss is −log({L.number(p, 2)}) = <span class="c-loss ea-num">{L.number(log, 2)}</span>, and the squared residual (1 − <i>ŷ</i>)<sup>2</sup> = <span class="c-loss ea-num">{L.number(sq, 2)}</span>.""")

    explore_p = L.explore("spam-p", "ŷ", _frame, start=0.5, low=0.01, high=0.99, step=0.01, role="pred", task="Move $\\hat{y}$ from 0.99 down to 0.01: where are the two losses close, and below which $\\hat{y}$ is the log loss more than 1?")
    explore_p
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Near $\hat{y} = 1$ the two are close. Below about $\hat{y} = 0.37$ the log loss is more than 1, and the squared residual never gets there. The log loss keeps climbing: at $\hat{y} = 0.01$ it is 4.61, against 0.98.

    ### Recipe

    To work out the log loss of a model on some examples:

    1. For each example, take the probability the model gave to what really happened: $\hat{y}$ if $y = 1$, $1 - \hat{y}$ if $y = 0$.
    2. Take minus its log: the surprise of each example.
    3. Take the mean over all examples: the log loss.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four mails: the first and the third are spam, the other two are not. The model gave them the probabilities of spam 0.9, 0.2, 0.6 and 0.7. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_log_loss():
    import numpy as np

    _y = np.array([1, 0, 1, 0])
    _p = np.array([0.9, 0.2, 0.6, 0.7])
    # 1. the probability of what really happened: p for spam, 1 - p for not spam
    _p_true = _y * _p + (1 - _y) * (1 - _p)
    print(_p_true)
    # 2. the surprise of each mail
    _surprise = -np.log(_p_true)
    print(_surprise)
    # 3. the mean: the log loss
    print(_surprise.mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_y * _p + (1 - _y) * (1 - _p)` picks $\hat{y}$ for the spam mails and $1 - \hat{y}$ for the others, because one of the two terms is always 0: 0.9, 0.8, 0.6 and 0.3. The surprises are 0.11, 0.22, 0.51 and 1.20, and their mean, the log loss, is 0.51. The fourth mail costs the most: the model gave 0.7 to spam, and it was not.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_not_spam():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A mail that is not spam
    """)
    return


@app.cell(hide_code=True)
def choice_not_spam_cell():
    choice_not_spam = L.choice(
        "not-spam",
        question="A mail is not spam, $y = 0$, and the model gave it the probability of spam 0.9. What is its log loss?",
        options={
            "spam": "$-\\log(0.9) \\approx 0.11$",
            "right": "$-\\log(0.1) \\approx 2.30$",
            "squared": "$0.9^2 = 0.81$",
            "negative": "$\\log(0.1) \\approx -2.30$",
        },
        correct="right",
        why={
            "spam": "That is the loss if the mail were spam. Which probability did the model give to what really happened? See step 1 of the recipe.",
            "squared": "That is the squared residual of the mean squared error. See the formula in *The idea*.",
            "negative": "A loss is never negative: the log of a probability below 1 is negative, so the loss is minus that. See step 2 of the recipe.",
            "right": "The model gave 1 − 0.9 = 0.1 to not spam, which is what happened. $-\\log(0.1)$ is about 2.30: confidently wrong.",
        },
    )
    choice_not_spam
    return (choice_not_spam,)


@app.cell(hide_code=True)
def text_exercise_one():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loss of one example

        Finish `one_loss(y, p)`: it gets the label `y`, 1 or 0, and `p`, the probability of class 1, and gives back the log loss of that one example. Replace the `...`. The check calls `one_loss(1, 0.9)`: that should give about `0.105`. It also tries other examples, with both labels.
        """),
        L.task("Make `one_loss` give back minus the log of the probability of what really happened."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one():
    box_one = L.code_box("one-loss", lines=8)
    box_one
    return (box_one,)


@app.cell(hide_code=True)
def check_one(box_one):
    result_one = L.check("one-loss", box_one.value)
    result_one
    return (result_one,)


@app.cell(hide_code=True)
def hints_one():
    L.hints(
        "one-loss",
        [
            "Steps 1 and 2 of the recipe, for one example: which probability belongs to what happened, then minus its log.",
            "Use an `if` on `y`. For `y == 1`, the model gave `p` to what happened. What did it give for `y == 0`?",
            "`if y == 1:` with `return -np.log(p)`, and under `else:` `return -np.log(___)`.",
        ],
        why="For `y == 1` the loss is `-np.log(p)`, for `y == 0` it is `-np.log(1 - p)`. For `one_loss(1, 0.9)`: −log(0.9) ≈ 0.105.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Five mails at once

        Five mails, with their labels in `y` and the model's probabilities of spam in `p`. Put their log loss in `loss`, in one line with the formula from *The idea*. Replace the `...`. The last line is `loss`, so the check sees it. The check also tries other probabilities.
        """),
        L.task("Put the mean log loss of the five mails in `loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-loss", lines=7)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-loss", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-loss",
        [
            "With arrays, `y * np.log(p)` works for all mails at once. What do you add for the mails that are not spam?",
            "The formula has two terms inside the mean, and a minus in front. Which NumPy function takes the mean?",
            "`loss = -np.mean(y * np.log(p) + ___)`.",
        ],
        why="`y * np.log(p) + (1 - y) * np.log(1 - p)` is, for each mail, the log of the probability of what happened; `-np.mean(...)` takes the mean with a minus in front: about 0.33.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [One number for how wrong: the loss](/course/open/ml-mse-loss?lang=en), the mean squared error measured how wrong a number was; the log loss measures how wrong a probability is, and punishes confident mistakes hardest. In [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), a model gave probabilities; now you can say how good they are. Later in the course, cross-entropy does the same for many classes, and a language model is trained on exactly this: the surprise of the next word. In the next lesson, *The gradient of log loss*, you work out how the log loss changes with each weight, so that training can lower it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_not_spam, result_mean, result_one):
    L.wrap_up(choice_not_spam, result_one, result_mean)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
