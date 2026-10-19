# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-log-loss-gradient. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-log-loss-gradient", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en), [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) and [Two parameters: the gradient](/course/open/math-gradient?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loss():
    quick_loss = L.choice(
        "quick-loss",
        question="A mail is spam, $y = 1$, and the model gives it the probability of spam 0.25. What is its log loss?",
        options={
            "other": "$-\\log(0.75) \\approx 0.29$",
            "right": "$-\\log(0.25) \\approx 1.39$",
            "squared": "$0.75^2 \\approx 0.56$",
        },
        correct="right",
        why={
            "other": "That is the loss of a mail that is not spam. Which probability did the model give to what really happened? See the recipe in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "squared": "That is the squared residual. See the formula in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "right": "The model gave 0.25 to spam, which is what happened: $-\\log(0.25) \\approx 1.39$.",
        },
    )
    quick_loss
    return (quick_loss,)


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="$u = 2x$ and $y = u^3$. What is $\\frac{dy}{dx}$ at $x = 1$?",
        options={
            "outer": "12",
            "inner": "2",
            "right": "24",
        },
        correct="right",
        why={
            "outer": "That is only the outer slope, $3u^2$ at $u = 2$. What do you multiply it by? See *The idea* in [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "inner": "That is only the inner slope. What do you multiply it by? See *The idea* in [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en).",
            "right": "The outer slope $3u^2 = 12$ at $u = 2$, times the inner slope 2: 24.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="Somewhere, the gradient of a loss is $(-2,\\ 0.5)$: $\\frac{\\partial \\mathcal{L}}{\\partial w} = -2$ and $\\frac{\\partial \\mathcal{L}}{\\partial b} = 0.5$. Which small change lowers the loss?",
        options={
            "right": "A bigger $w$ and a smaller $b$",
            "both-up": "A bigger $w$ and a bigger $b$",
            "uphill": "A smaller $w$ and a bigger $b$",
        },
        correct="right",
        why={
            "both-up": "A positive slope means the loss rises as $b$ grows. Which way lowers it? See *The idea* in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "uphill": "That is the direction of the gradient itself: uphill. See *The idea* in [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "right": "Go against each slope: $w$ up, because its slope is negative, and $b$ down, because its slope is positive.",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To train logistic regression with gradient descent, you need the slope of the log loss for each weight and for the bias term. Nudging each one works, but a model with a million weights would need a million nudges per step. The chain rule gives every slope from one formula. For log loss with a sigmoid, that formula turns out surprisingly simple: the probability minus the label, times the feature.

    **An everyday example.** To know what a taxi ride costs, you do not take every ride to find out. The fare, a start price plus a price per kilometre, gives the cost of every ride at once.
    """)
    return


@app.cell(hide_code=True)
def predict_correction():
    guess_correction = L.guess(
        "correction",
        question="Two spam mails, with the same features. The model gives one the probability of spam 0.9, the other 0.1. For which one do you think the log loss changes faster when a weight changes?",
        options={
            "ninety": "The mail with 0.9",
            "ten": "The mail with 0.1",
            "same": "Both the same",
        },
        reveal="**The mail with 0.1.** The slope of the log loss for a weight turns out to be $(\\hat{y} - y) \\cdot x$: the probability minus the label, times the feature. For 0.1 that is $0.1 - 1 = -0.9$ times the feature, for 0.9 only $-0.1$ times it: nine times smaller. The mail the model is far off on gets the big correction.",
    )
    guess_correction
    return (guess_correction,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The log loss of one example is built in three steps: the logit $z = \mathbf{w} \cdot \mathbf{x} + b$, the probability $\hat{y} = \sigma(z)$, and the loss $\mathcal{L} = -\left( y \log \hat{y} + (1 - y) \log(1 - \hat{y}) \right)$. The chain rule multiplies the slopes of the steps, now three instead of two. Two of these slopes you take as given here: those of $\log$ and of $\sigma$ follow from rules for exp and log that a later lesson derives.

        - **The slope of the loss for $\hat{y}$.** The slope of $\log(p)$ is $1/p$. In $\log(1 - \hat{y})$ the inner step $1 - \hat{y}$ has the slope $-1$, so by the chain rule its slope is $-\frac{1}{1 - \hat{y}}$. With the minus in front of the loss, the slope of $\mathcal{L}$ is $-\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$.
        - **The slope of the sigmoid.** The slope of $\sigma(z)$ is $\sigma(z)\,(1 - \sigma(z))$, here $\hat{y}(1 - \hat{y})$.
        - **The slope of the logit for a weight.** $z$ is a dot product, so a weight $w_j$ that grows by 1 makes $z$ grow by $x_j$: the slope is $x_j$. For the bias term it is 1.

        Multiply the first two, and the fractions cancel:
        """),
        L.formula(r"\left(-\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}\right) \hat{y}(1 - \hat{y}) = -y(1 - \hat{y}) + (1 - y)\hat{y}"),
        mo.md(r"""
        Work out the brackets, and the two terms with $y\hat{y}$ cancel too:
        """),
        L.formula(r"-y + y\hat{y} + \hat{y} - y\hat{y} = \hat{y} - y"),
        mo.md(r"""
        With the third:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] = ([[pred:\hat{y}]] - y)\, x_j \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b}]] = [[pred:\hat{y}]] - y"),
        mo.md(r"""
        For many examples the log loss is a mean, so each slope is the mean of these slopes over the examples. The slider below checks the slope of $\sigma$ by nudging, as in [The derivative as a slope](/course/open/math-slope?lang=en); the worked example checks the whole formula.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **derivative of the sigmoid**, $\sigma(z)(1 - \sigma(z))$ | sigma of z times one minus sigma of z | the slope of the sigmoid at $z$, how fast it rises there: 0.25 at $z = 0$, close to 0 far from it | `p * (1 - p)` |
        | **gradient of the log loss**, $(\hat{y} - y)\, x_j$ | y hat minus y, times x j | the slope of the log loss of logistic regression for the weight $w_j$: the probability minus the label, times the feature | `(p - y) * x` |
        """),
        L.watch_out("The slope of the sigmoid cancels against the slope of the log; that is why the gradient is so simple. With the mean squared error on a probability, it stays in: a model that is confidently wrong, with $\\hat{y}(1 - \\hat{y})$ close to 0, would then barely learn."),
        mo.md(r"""
        Move the slider, or drag across the chart: it sets the logit $z$. The <span class="c-grad">green line</span> has the slope of the sigmoid at that point, $\hat{y}(1 - \hat{y})$.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        p = C.sigmoid(z)
        slope = p * (1 - p)
        nudged = (C.sigmoid(z + 0.001) - p) / 0.001
        return (C.picture(z, {"title": "The slope of the sigmoid", "z": "logit z", "p": "probability ŷ",
                              "alt": f"The sigmoid, with a point at z = {L.number(z, 1)} and a short green line with the slope there, {L.number(slope, 3)}. The sigmoid is steepest at z = 0 and almost flat far to the left and right."}),
                f"""At <i>z</i> = {L.number(z, 1)}: <i>ŷ</i> = <span class="c-pred ea-num">{L.number(p, 3)}</span>, and the slope <i>ŷ</i>(1 − <i>ŷ</i>) = <span class="c-grad ea-num">{L.number(slope, 4)}</span>. Nudging <i>z</i> by 0.001 gives <span class="c-grad ea-num">{L.number(nudged, 4)}</span>.""")

    explore_z = L.explore("slope-z", "z", _frame, start=0, low=-5, high=5, step=0.5, role="", task="How many times smaller is the slope $\\hat{y}(1 - \\hat{y})$ at $z = -5$ than at $z = 0$?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    At $z = -5$ the slope is 0.0066: about 38 times smaller than 0.25 at $z = 0$, and nudging gives almost the same, 0.0067. A spam mail with that logit gets $\hat{y} = 0.007$: the model is confidently wrong. Log loss still gives it the slope $\hat{y} - y \approx -0.99$ for the bias term, almost the biggest there is; with the mean squared error, this slope of the sigmoid, 0.0066, stays in its gradient, and the model would barely learn from this mail.

    ### Recipe

    To work out the gradient of the log loss of logistic regression:

    1. Work out the probability $\hat{y}$ of each example: the sigmoid of the logit.
    2. Subtract the label from each probability: $\hat{y} - y$.
    3. The slope for each weight is the mean of $\hat{y} - y$ times its feature; for the bias term, the mean of $\hat{y} - y$ itself.
    4. Check one slope by nudging: change that weight by a tiny $h$, and divide the change in the loss by $h$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One spam mail, $y = 1$, with the features 2 and 1, and a model with the weights 0.5 and −1 and the bias term 0. First the gradient from the formula, then the check by nudging the first weight. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_gradient():
    import numpy as np

    _x = np.array([2, 1])
    _y = 1
    _w = np.array([0.5, -1])
    _b = 0

    def _loss(w, b):
        _p = 1 / (1 + np.exp(-(np.dot(w, _x) + b)))
        return -(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

    # 1. the probability
    _p = 1 / (1 + np.exp(-(np.dot(_w, _x) + _b)))
    print(_p)
    # 2. the probability minus the label
    print(_p - _y)
    # 3. the slopes for the two weights, and for the bias term
    print((_p - _y) * _x, _p - _y)
    # 4. the check: nudge the first weight by h
    _h = 0.001
    print((_loss(_w + np.array([_h, 0]), _b) - _loss(_w, _b)) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The logit is $0.5 \cdot 2 - 1 \cdot 1 + 0 = 0$, so $\hat{y} = 0.5$ and $\hat{y} - y = -0.5$. The slopes are $-0.5 \cdot 2 = -1$ and $-0.5 \cdot 1 = -0.5$ for the weights, and $-0.5$ for the bias term; with one example, the mean of step 3 is that one value. Nudging the first weight gives $-0.9995$: close to $-1$, so the formula passes the check. All slopes are negative: raise the weights and the bias term a little, and the loss of this spam mail goes down.
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
        "not-spam-slope",
        question="A mail is not spam, $y = 0$. The model gives it $\\hat{y} = 0.8$, and its first feature is $x_1 = 3$. What is $\\frac{\\partial \\mathcal{L}}{\\partial w_1}$ for this mail?",
        options={
            "spam": "$-0.6$",
            "no-x": "$0.8$",
            "right": "$2.4$",
            "sign": "$-2.4$",
        },
        correct="right",
        why={
            "spam": "That is $(0.8 - 1) \\cdot 3$, as if the mail were spam. What is $y$ here? See the formula in *The idea*.",
            "no-x": "That is $\\hat{y} - y$ alone, the slope for the bias term. What does the weight of feature 1 add? See the formula in *The idea*.",
            "sign": "The formula has $\\hat{y} - y$, the probability minus the label, not $y - \\hat{y}$. See the formula in *The idea*.",
            "right": "$(0.8 - 0) \\cdot 3 = 2.4$: positive, so a smaller $w_1$ lowers the loss. The model gave too much to spam.",
        },
    )
    choice_not_spam
    return (choice_not_spam,)


@app.cell(hide_code=True)
def text_exercise_one():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The gradient of one example

        Finish `one_gradient(x, y, w, b)`. It gets the features `x` of one example, its label `y`, the weights `w` and the bias term `b`, and gives back the slopes of the log loss for the weights, as an array. The box already works out the probability `p`. Replace the `...`. The check calls `one_gradient([2, 1], 1, [0.5, -1], 0)`, the mail of the worked example: that should give `array([-1. , -0.5])`. It also tries other examples.
        """),
        L.task("Make `one_gradient` give back the slopes of the log loss for the weights."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one():
    box_one = L.code_box("one-gradient", lines=8)
    box_one
    return (box_one,)


@app.cell(hide_code=True)
def check_one(box_one):
    result_one = L.check("one-gradient", box_one.value)
    result_one
    return (result_one,)


@app.cell(hide_code=True)
def hints_one():
    L.hints(
        "one-gradient",
        [
            "The formula in *The idea*: the probability minus the label, times the features.",
            "With arrays, one multiplication gives every slope at once. Which order: `p - y` or `y - p`?",
            "`return (p - ___) * ___`.",
        ],
        why="`(p - y) * x` multiplies `p - y` by every feature at once. For the mail: $-0.5 \\cdot [2, 1] = [-1, -0.5]$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The gradient for all students

        The students of [Examples, features and labels](/course/open/ml-vocabulary?lang=en) whose result is known, with the model of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en): the weights 1 and 0.5, and the bias term −8. The box works out the probability `p` of every student. Put the three slopes of the mean log loss in `grad`, as one array: for the weight of `hours`, the weight of `sleep`, and the bias term. Replace the `...`. The last line is `grad`, so the check sees it. The check also tries another bias term.
        """),
        L.task("Put the gradient of the mean log loss in `grad`: three slopes in one array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-gradient", lines=11)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-gradient", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-gradient",
        [
            "Steps 2 and 3 of the recipe, for all students at once: first the probability minus the label, then a mean per slope.",
            "`p - passed` gives one number per student. Which feature belongs to which slope, and what does the bias term get?",
            "`grad = np.array([np.mean((p - passed) * hours), np.mean(___), np.mean(___)])`.",
        ],
        why="`p - passed` is the probability minus the label of every student; times `hours` and `sleep`, and averaged, it gives the slopes of the two weights; the mean of `p - passed` itself is the slope of the bias term: about −0.33, −0.22 and −0.04.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Rates multiply: the chain rule](/course/open/math-chain-rule?lang=en) you multiplied the slopes of the steps of one function; here the steps are the logit, the sigmoid and the log, and the slope of the sigmoid cancels. In [Two parameters: the gradient](/course/open/math-gradient?lang=en) a slope per parameter made the gradient; now you have a formula for it, checked by nudging. In [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en) you measured how wrong the probabilities were; now you know which way to change the model. In the next lesson, *Training logistic regression*, gradient descent uses exactly this gradient to train the model.
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
