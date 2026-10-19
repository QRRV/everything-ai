# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-regularization. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-regularization", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) and [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_choose():
    quick_choose = L.choice(
        "quick-choose",
        question="On which set do you choose a hyperparameter?",
        options={
            "train": "The training set",
            "right": "The validation set",
            "test": "The test set",
        },
        correct="right",
        why={
            "train": "On the training set more capacity always looks better. See *The idea* in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "test": "The test set stays sealed until the end, for one honest score. See *The idea* in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "right": "Train with each value on the training set, and keep the one with the lowest validation loss.",
        },
    )
    quick_choose
    return (quick_choose,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="$w = 3$ and $b = 0$. The gradient there is $(2,\\ -1)$, and the learning rate is 0.5. Where are $w$ and $b$ after one update step?",
        options={
            "uphill": "$w = 4$, $b = -0.5$",
            "half": "$w = 2$, $b = -0.5$",
            "right": "$w = 2$, $b = 0.5$",
        },
        correct="right",
        why={
            "uphill": "That is a step along the gradient, uphill. See the recipe in [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en).",
            "half": "Check the sign for $b$: its slope is negative. See the recipe in [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en).",
            "right": "$w = 3 - 0.5 \\cdot 2 = 2$ and $b = 0 - 0.5 \\cdot (-1) = 0.5$: against each slope.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model with many features and few examples overfits: it gives large weights to features that only happen to fit the noise of its training examples. You could take features away by hand, but you often do not know which ones. Instead, make large weights cost something: add a penalty for them to the loss. Gradient descent then only lets a weight grow where that clearly lowers the loss.

    **An everyday example.** An airline charges for every extra kilo of luggage. You still take what you really need, but you pack much less of what is just in case.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights",
        question="Twelve examples with eight features each; the labels depend only on the first two. Plain linear regression fits the twelve examples almost exactly. What do you think its weights for the other six features look like?",
        options={
            "zero": "Close to 0: those features do not matter",
            "some": "Some of them far from 0",
            "same": "All about as big as the first two",
        },
        reveal="**Some of them far from 0.** In the worked example below, feature 3 gets a weight of about −1.9, although it has nothing to do with the labels. With eight features and only twelve examples, the model uses them to fit the noise. On new examples that costs it: an MSE of about 1.28 on the validation set.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Regularization** is any change to training that keeps a model simpler, so that it generalizes better. The most common kind, **L2 regularization**, adds the sum of the squared weights to the loss, times a number $\lambda$:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = \frac{1}{n} \sum_{i=1}^{n} (y_i - [[pred:\hat{y}_i]])^2 + [[hyper:\lambda]] \sum_{j} [[param:w_j]]^2"),
        mo.md(r"""
        Large weights now cost something, so gradient descent only lets a weight grow where that clearly lowers the first part. The bias term is left out of the penalty: it only shifts every prediction by the same amount. $\lambda$ is a hyperparameter: with $\lambda = 0$ you have plain linear regression; with a very large $\lambda$ every weight goes to almost 0. You choose it on the validation set.

        The slope of $\lambda w_j^2$ for $w_j$ is $2 \lambda w_j$, so the slope for each weight gets $2 \lambda w_j$ extra. One update step then is $w_j - \eta\,(\text{slope of the MSE} + 2 \lambda w_j) = (1 - 2 \eta \lambda)\, w_j - \eta \cdot \text{slope of the MSE}$: before the usual step, every weight shrinks by the same factor. That is why it is also called **weight decay**. For a small $\lambda$ that factor is just below 1; with a large $\lambda$, choose a smaller learning rate, so that $2 \eta \lambda$ stays well below 1.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **regularization** | | any change to training that keeps a model simpler, so that it generalizes better | |
        | **L2 regularization** | L two | adding $\lambda$ times the sum of the squared weights to the loss | `lam * np.sum(w ** 2)` |
        | $\lambda$ | lambda | how heavy the penalty is: a hyperparameter | `lam` |
        | **weight decay** | | every update step shrinks each weight a little toward 0; for gradient descent the same as L2 regularization | `2 * lam * w` in the slope |
        """),
        L.watch_out("Choose $\\lambda$ on the validation set, never on the training set. On the training set, $\\lambda = 0$ always wins: the penalty only makes the fit to the training examples worse."),
        mo.md(r"""
        Move the slider, or drag across the chart of the MSE: it sets $\lambda$. One chart shows the eight <span class="c-param">weights</span>; the dashes mark the weights the labels were made with, 3, −2 and six 0s. The other shows the <span class="c-loss">MSE</span> for every $\lambda$: on the training set (solid line, filled dot) and on the validation set (dashed line, open dot). The picture works out the weights directly, without steps: the ones gradient descent ends at after very many steps. That way it follows the slider at once.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lambda_cell():
    def _frame(lam):
        lam = round(lam, 2)
        train, val = C.errors(lam)
        w, _ = C.fit(lam)
        return (C.picture(lam, {"weights": "The eight weights", "feature": "feature", "weight": "weight",
                                "mse": "MSE per λ", "loss": "MSE",
                                "alt": f"Eight bars, one per weight, with λ = {L.number(lam, 2)}: the first {L.number(w[0], 2)}, the second {L.number(w[1], 2)}, the other six between {L.number(min(w[2:]), 2)} and {L.number(max(w[2:]), 2)}. The MSE on the training set rises with λ; on the validation set it falls steeply from 1.28 to about 0.15 between λ = 0 and 0.01, and then rises again."}),
                f"""With <i>λ</i> = {L.number(lam, 2)}: an MSE of <span class="c-loss ea-num">{L.number(train, 3)}</span> on the training set and <span class="c-loss ea-num">{L.number(val, 3)}</span> on the validation set.""")

    explore_lambda = L.explore("lambda", "λ", _frame, start=0, low=0, high=0.2, step=0.01, role="hyper", task="Find the $\\lambda$ with the lowest MSE on the validation set. What happens to the first two weights when $\\lambda$ gets bigger than that?")
    explore_lambda
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    On the validation set the MSE is lowest around $\lambda = 0.01$. The dashed curve dips to about 0.15 at a $\lambda$ just below 0.01; of the slider's stops, 0.01 gives the lowest MSE, 0.20. A larger $\lambda$ shrinks the first two weights as well, the ones that matter, and the MSE on the validation set rises again: underfitting.

    ### Recipe

    To train with L2 regularization:

    1. Add $\lambda$ times the sum of the squared weights to the loss: `lam * np.sum(w ** 2)`.
    2. In the slopes, add `2 * lam * w` to the slopes for the weights; leave the slope for the bias term alone.
    3. Train with gradient descent as before.
    4. Try a few values of $\lambda$, and choose on the validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Forty made-up examples with eight features between 0 and 1. The labels are $1 + 3 \cdot \text{feature 1} - 2 \cdot \text{feature 2}$, in code `_X[:, 0]` and `_X[:, 1]`, plus noise of at most 0.5 up or down; the other six features have nothing to do with them. The examples were drawn at random, so the first 12 can be the training set and the other 28 the validation set. Gradient descent runs 20,000 steps with $\eta$ = 0.3, once with $\lambda = 0$ and once with $\lambda = 0.01$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_regularization():
    import numpy as np

    _rng = np.random.default_rng(2)
    _X = _rng.random(size=(40, 8))
    _y = 1 + 3 * _X[:, 0] - 2 * _X[:, 1] + (_rng.random(size=40) - 0.5)
    _X_train = _X[:12]
    _y_train = _y[:12]
    _X_val = _X[12:]
    _y_val = _y[12:]

    for _lam in [0, 0.01]:
        _w = np.zeros(8)
        _b = 0.0
        for _step in range(20000):
            _r = _y_train - (_X_train @ _w + _b)
            # 1. the regularized loss: the MSE plus the penalty
            _loss = np.mean(_r ** 2) + _lam * np.sum(_w ** 2)
            # 2. its slopes: 2 * lam * w extra, for the weights only
            _grad_w = -2 * (_r @ _X_train) / len(_y_train) + 2 * _lam * _w
            _grad_b = -2 * _r.mean()
            # 3. one step of gradient descent
            _w = _w - 0.3 * _grad_w
            _b = _b - 0.3 * _grad_b
        # 4. the weights, the loss, and the MSE on the validation set
        print(_lam, _w)
        print(_loss)
        print(np.mean((_y_val - (_X_val @ _w + _b)) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With $\lambda = 0$ the weights are about 3.64, −3.27, −1.91, 0.84 and so on: feature 3 gets −1.91 although it does not matter. The loss on the training set is only about 0.015, but the MSE on the validation set is about 1.28. With $\lambda = 0.01$ the first two weights shrink to about 2.28 and −1.60, and the other six all stay within about 0.61 of 0. The loss on the training set rises to about 0.15, about half of it penalty, but the MSE on the validation set falls to about 0.20.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_big():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A very large λ
    """)
    return


@app.cell(hide_code=True)
def choice_big_cell():
    choice_big = L.choice(
        "big-lambda",
        question="What happens if you train with a very large $\\lambda$, such as 100?",
        options={
            "best": "The model gets better and better on new data",
            "right": "All weights end up close to 0, and the model underfits",
            "nothing": "Nothing changes: the penalty does not touch the predictions",
        },
        correct="right",
        why={
            "best": "Move the slider to the right in *The idea*: what does the validation MSE do?",
            "nothing": "The penalty changes the weights, and the weights make the predictions. See *The idea*.",
            "right": "The penalty then outweighs everything: the weights shrink to almost 0, and the model predicts about the same for every example.",
        },
    )
    choice_big
    return (choice_big,)


@app.cell(hide_code=True)
def text_exercise_penalty():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The regularized loss

        Four examples with two features, a model with the weights `w` and the bias term `b`, and `lam`, the $\lambda$ of the penalty. Put the regularized loss in `loss`: the mean squared error, plus `lam` times the sum of the squared weights. Replace the `...`. The last line is `loss`, so the check sees it. The check also tries other weights.
        """),
        L.task("Put the mean squared error plus the L2 penalty in `loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_penalty():
    box_penalty = L.code_box("penalty-loss", lines=9)
    box_penalty
    return (box_penalty,)


@app.cell(hide_code=True)
def check_penalty(box_penalty):
    result_penalty = L.check("penalty-loss", box_penalty.value)
    result_penalty
    return (result_penalty,)


@app.cell(hide_code=True)
def hints_penalty():
    L.hints(
        "penalty-loss",
        [
            "Step 1 of the recipe: the loss you already know, with the penalty added.",
            "The predictions are `X @ w + b`. Which weights go into the penalty, and how do you add up their squares?",
            "`loss = np.mean((y - (X @ w + b)) ** 2) + ___`.",
        ],
        why="The mean squared error is about 0.22. The sum of the squared weights is $2^2 + (-1.5)^2 = 6.25$, so the penalty is 0.1 times 6.25 = 0.625. Together about 0.85.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The slopes with weight decay

        Finish `decay_gradients(X, y, w, b, lam)`: it gives back the slopes of the regularized loss, `grad_w` for the weights and `grad_b` for the bias term. The residuals `r` are already there. Replace the `...`. The check calls it with the four examples of exercise 2: that should give about `([-0.0225, -0.7825], -0.925)`. It also tries other examples, weights and values of `lam`.
        """),
        L.task("Make `decay_gradients` give back the slopes of the regularized loss."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("decay-gradients", lines=10)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("decay-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "decay-gradients",
        [
            "Step 2 of the recipe: the slopes of the mean squared error, as in linear regression, plus the slope of the penalty for the weights.",
            "The slopes of the mean squared error are `-2 * (r @ X) / len(y)` and `-2 * r.mean()`. What is the slope of `lam * w ** 2`, and which of the two gets it?",
            "`grad_w = -2 * (r @ X) / len(y) + ___` and `grad_b = -2 * r.mean()`.",
        ],
        why="The slopes of the mean squared error are about $(-0.4225,\\ -0.4825)$ and $-0.925$. The penalty adds $2 \\cdot 0.1 \\cdot (2,\\ -1.5) = (0.4,\\ -0.3)$ to the first two, giving $(-0.0225,\\ -0.7825)$; the slope for $b$ stays $-0.925$.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en) each step went against the slopes of the loss; with weight decay, the slopes get $2 \lambda w$ extra, and every step also pulls the weights toward 0. In [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) you chose a degree on the validation set; $\lambda$ is chosen the same way. In the next lesson, *The loss you train on and the metric you report*, you see why the number you train on is not always the number you tell others.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_big, result_penalty, result_gradients):
    L.wrap_up(choice_big, result_penalty, result_gradients)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
