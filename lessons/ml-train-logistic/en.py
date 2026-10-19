# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-train-logistic. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-train-logistic", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en) and [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="A spam mail, $y = 1$, gets $\\hat{y} = 0.7$ from the model, and its feature is $x = 2$. What is the slope of its log loss for the weight of that feature?",
        options={
            "right": "$-0.6$",
            "sign": "$0.6$",
            "no-x": "$-0.3$",
        },
        correct="right",
        why={
            "sign": "The gradient starts from $\\hat{y} - y$, the probability minus the label, not $y - \\hat{y}$. See the formula in [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en).",
            "no-x": "That is $\\hat{y} - y$ alone, the slope for the bias term. What does the weight of a feature add? See the formula in [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en).",
            "right": "$(0.7 - 1) \\cdot 2 = -0.6$: negative, so a bigger weight lowers the loss of this mail.",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="What does one round of the loop in your `fit` for linear regression do?",
        options={
            "predict": "It predicts the labels once more, and keeps the best weights",
            "nudge": "It nudges each weight and measures the loss again",
            "right": "It gets both slopes from `mse_gradients`, and moves `w` and `b` against them",
        },
        correct="right",
        why={
            "predict": "Nothing is kept or compared: each round changes `w` and `b`. How? See exercise 3 of [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "nudge": "Nudging was the slow way of [Fitting w and b with gradient descent](/course/open/ml-gradient-descent-2d?lang=en). Where does `fit` get its slopes? See exercise 3 of [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en).",
            "right": "`grad_w, grad_b = gradients.mse_gradients(X, y, w, b)`, then `w = w - lr * grad_w` and `b = b - lr * grad_b`, `steps` times.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        You have every piece of a classifier, a model for classification, that learns: the sigmoid of a dot product gives a probability, log loss says how wrong it is, and the gradient $(\hat{y} - y)\,x$ says which way to change each weight. In [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en), gradient descent trained a linear model. Now it trains logistic regression, and the loop stays exactly the same: only the slopes it gets are different. That is the second step of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           1, here_label="← today", title="Where your project stands: your own neural network"),
        mo.md(r"""
        **An everyday example.** A bread recipe also works for rolls: the same steps, kneading, rising, baking, with other amounts and another time in the oven. You do not learn to bake all over again.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_change():
    guess_change = L.guess(
        "change",
        question="Your `fit` trained linear regression with the slopes of the MSE. To train logistic regression instead, what do you think has to change?",
        options={
            "all": "Almost all of it",
            "slopes": "Only where the slopes come from",
            "nothing": "Nothing at all",
        },
        reveal="**Only where the slopes come from.** The loop starts with zeros, gets both slopes and steps against them, `steps` times: that is gradient descent, whatever the model. For logistic regression the slopes are those of the log loss: the probabilities through the sigmoid, the differences $\\hat{y} - y$, and their means. Exercise 2 writes that function; exercise 3 puts it in the loop.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Gradient descent does not care which model it trains: it needs the slopes of the loss for every parameter, and steps against them. For logistic regression on $n$ examples at once, the slopes of the log loss are the means of $(\hat{y} - y)\,x_j$ and of $\hat{y} - y$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_i]] - y_i) \, x_{ij} \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_i]] - y_i)"),
        mo.md(r"""
        The sum over $i$ multiplies each difference $\hat{y}_i - y_i$ by the number in the same place of column $j$ of $\mathbf{X}$ and adds up: a dot product. For all weights at once, that is `(p - y) @ X`, as `r @ X` was for linear regression. Side by side:

        | | Linear regression | Logistic regression |
        |---|---|---|
        | prediction | `X @ w + b` | `1 / (1 + np.exp(-(X @ w + b)))` |
        | loss | mean squared error | log loss |
        | slopes for `w` | `-2 * (r @ X) / n`, with `r = y - y_pred` | `(p - y) @ X / n` |
        | slope for `b` | `-2 * r.mean()` | `(p - y).mean()` |
        | one step | `w = w - lr * grad_w`, `b = b - lr * grad_b` | the same |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **training with gradient descent** | | starting from zeros, and stepping against the slopes of the loss, again and again, until the loss is low: for any model whose slopes you can work out | `for step in range(steps):` |
        """),
        L.watch_out("Mind the sign. For log loss the gradient starts from $\\hat{y} - y$, prediction minus label, with no factor in front. In your linear regression the residual was $y - \\hat{y}$, label minus prediction, with $-2$ in front. Mix them, and gradient descent walks uphill."),
        mo.Html(C.curve_picture({"title": "The log loss of the students during training", "x": "step", "y": "log loss",
                                 "alt": "The log loss of the seven students during 1000 steps of gradient descent with a learning rate of 0.1: it starts at 0.69, drops fast in the first hundred steps, and creeps towards 0 after that."})),
        mo.md(r"""
        Here the course trains on the seven students of [Examples, features and labels](/course/open/ml-vocabulary?lang=en), with $\eta$ = 0.1. The <span class="c-loss">log loss</span> starts at 0.69, the loss of a probability of 0.5 for everyone, and falls. After exercise 3, the demo trains your own `fit` the same way.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One step of gradient descent for logistic regression:

    1. Predict a probability for all examples at once: `p = 1 / (1 + np.exp(-(X @ w + b)))`.
    2. Work out the differences between probability and label: `diff = p - y`.
    3. Work out every slope: `grad_w = (diff @ X) / n` and `grad_b = diff.mean()`.
    4. Step against the slopes, `w = w - lr * grad_w` and `b = b - lr * grad_b`, and repeat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The seven students: hours of study and of sleep in `X`, whether they passed in `y`. One step from $\mathbf{w} = 0$ and $b = 0$, with $\eta$ = 0.1, and the log loss before and after. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_step():
    import numpy as np

    _X = np.array(C.STUDENTS)
    _y = np.array(C.PASSED)
    _w = np.zeros(2)
    _b = 0.0

    def _loss(w, b):
        _p = 1 / (1 + np.exp(-(_X @ w + b)))
        return -np.mean(_y * np.log(_p) + (1 - _y) * np.log(1 - _p))

    # 1. a probability for every student
    _p = 1 / (1 + np.exp(-(_X @ _w + _b)))
    print(_p)
    # 2. the differences between probability and label
    _diff = _p - _y
    print(_diff)
    # 3. every slope at once
    _grad_w = (_diff @ _X) / len(_y)
    _grad_b = _diff.mean()
    print(_grad_w, _grad_b)
    # 4. one step, and the loss before and after
    _lr = 0.1
    print(_w - _lr * _grad_w, _b - _lr * _grad_b)
    print(_loss(_w, _b), _loss(_w - _lr * _grad_w, _b - _lr * _grad_b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With all weights 0, every logit is 0 and every probability 0.5: the model does not know anything yet. The differences are 0.5 for the three students who failed and −0.5 for the four who passed. The slopes are about −1.57 for the hours of study, −0.71 for sleep and −0.07 for $b$: all negative, so both weights and $b$ go up. One step brings the log loss down from 0.69 to 0.60. In exercise 3 your `fit` repeats this step in a loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One step by hand
    """)
    return


@app.cell(hide_code=True)
def choice_step_cell():
    choice_step = L.choice(
        "one-step",
        question="The weights are $\\mathbf{w} = (1, 1)$, the learning rate is 0.5, and the slopes for the weights are $(-0.4,\\ 0.2)$. What are the weights after one step?",
        options={
            "uphill": "$(0.8,\\ 1.1)$",
            "no-lr": "$(1.4,\\ 0.8)$",
            "right": "$(1.2,\\ 0.9)$",
        },
        correct="right",
        why={
            "uphill": "That steps with the slopes, uphill. Which way does a step of gradient descent go? See step 4 of the recipe.",
            "no-lr": "That steps by the whole slope. What does the learning rate do? See step 4 of the recipe.",
            "right": "$1 - 0.5 \\cdot (-0.4) = 1.2$ and $1 - 0.5 \\cdot 0.2 = 0.9$: against each slope, by half of it.",
        },
    )
    choice_step
    return (choice_step,)


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The slopes of log loss

        Finish `log_loss_gradients(X, y, w, b)`. It gets the examples `X`, the labels `y`, 1 or 0, the weights `w` and the bias term `b`, and gives back the slopes of the log loss: `grad_w`, one per weight, and `grad_b`. Replace the `...` with steps 1 to 3 of the recipe. The check calls `log_loss_gradients([[2, 1], [1, 3], [0, 1]], [1, 0, 1], [0, 0], 0)`: that should give about `([-0.1667, 0.1667], -0.1667)`. It also tries other examples, weights and bias terms. When your function is right, the check saves it in your project, as `work/projects/nn/logistic_gradients.py`.
        """),
        L.task("Make `log_loss_gradients` give back the slopes of the log loss for every weight and for `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("log-loss-gradients", lines=11)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("log-loss-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "log-loss-gradients",
        [
            "Steps 1 to 3 of the recipe, in the same order: the probabilities, the differences, then the two slopes.",
            "The differences are the probabilities minus the labels. In which order do you multiply the differences and `X`, and what do you divide by?",
            "`p = 1 / (1 + np.exp(-(X @ w + b)))`, `diff = p - y`, `grad_w = (___) / len(y)` and `grad_b = ___`.",
        ],
        why="With `w` and `b` all 0, every probability is 0.5, so the differences are `[-0.5, 0.5, -0.5]`. `diff @ X` is `[-0.5, 0.5]`; divided by 3, that is about `[-0.1667, 0.1667]`, and the mean of the differences is about −0.1667.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Train

        Now your project trains a classifier. `fit(X, y, lr, steps)` imports your `log_loss_gradients` from your project, starts with all weights 0 and $b$ = 0, and runs its loop `steps` times: the same as your `fit` for linear regression. Replace the `...` with one update step: get both slopes with `logistic_gradients.log_loss_gradients(X, y, w, b)`, and move `w` and `b` against them. The check calls `fit([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 0, 1, 0], 0.5, 200)`: there, only the first feature tells the classes apart. It also tries other data, learning rates and numbers of steps. When your function is right, the check saves it in your project, as `work/projects/nn/logistic.py`.
        """),
        L.task("Make `fit` train `w` and `b` with your own `log_loss_gradients`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-logistic", lines=13)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit, result_gradients):
    result_gradients  # check again when exercise 2 saves your module
    result_fit = L.check("fit-logistic", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-logistic",
        [
            "Step 4 of the recipe, with the slopes from your own function: the same loop as in your `fit` for linear regression.",
            "The function lives in the module `logistic_gradients`: call it through the module. It gives back two things at once: how do you give each its own name?",
            "`grad_w, grad_b = logistic_gradients.log_loss_gradients(X, y, w, b)`, then `w = w - ___` and `b = b - ___`.",
        ],
        why="`logistic_gradients.log_loss_gradients(X, y, w, b)` gives both slopes; `grad_w, grad_b = ...` catches them. `w = w - lr * grad_w` and `b = b - lr * grad_b` step against them. After 200 rounds on the example, `w` is about `[5.89, -0.42]` and `b` about −2.52.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your own `fit` to train logistic regression on the seven students, with $\eta$ = 0.1 and 1000 steps, and draws its decision boundary. It uses your own module, so it works once exercise 3 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_students(result_fit):
    result_fit  # run again when exercise 3 saves your module
    _model = C.trained()
    if _model is None:
        _out = mo.md("*Finish exercise 3 first: then your `fit` is in your project, and this cell uses it.*")
    elif _model == "gradients":
        _out = mo.md("*Your `fit` gets its slopes from `log_loss_gradients` in `work/projects/nn/logistic_gradients.py`, and that file is missing, or no longer gives the right slopes. Check exercise 2 again; if you changed that file, delete it first.*")
    elif not _model:
        _out = mo.md("*The `fit` in `work/projects/nn/logistic.py` no longer trains right, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*")
    else:
        _w, _b = _model
        _out = mo.vstack([
            mo.md(f"""Your model learned the weights <span class="c-param ea-num">{L.number(_w[0], 2)}</span> for the hours of study and <span class="c-param ea-num">{L.number(_w[1], 2)}</span> for the hours of sleep, and the bias term <span class="c-param ea-num">{L.number(_b, 2)}</span>. It puts <span class="ea-num">{C.wrong_side(_w, _b)}</span> of the 7 students on the wrong side of its decision boundary."""),
            mo.Html(C.demo_picture(_w, _b, {"title": "Your model on the seven students", "hours": "hours of study", "sleep": "hours of sleep",
                                            "alt": "The seven students by hours of study and sleep, four filled dots that passed and three open dots that failed, with the decision boundary of your trained model as a steep blue line between them."})),
            mo.md("""The filled dots passed, the open dots failed. Look at the weight for sleep: it is negative, although the students who passed slept a little more. That says little about sleep. Every student sleeps 5 to 8 hours, so a negative weight for sleep lowers every logit by about the same amount, as a negative bias term would. With 3.57 per hour of study, every logit needs such a large negative part, so that the students with few hours of study end up below 0. Gradient descent gets it mostly from sleep: the slope for that weight multiplies each difference by 5 to 8 hours of sleep before it takes the mean, and the slope for b is only the mean of the differences. After 1000 steps, the weight for sleep times 7 hours is about −12, and b is only −0.55. And on data it can split perfectly, like these seven students, the weights keep growing the longer it trains, while the log loss creeps towards 0."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The gradient of log loss](/course/open/ml-log-loss-gradient?lang=en) you derived the slopes $(\hat{y} - y)\,x$; in [Linear regression on many features](/course/open/ml-linear-regression-numpy?lang=en) you wrote the loop that steps against slopes. Put together, they train a classifier, and your project `nn` now has `logistic_gradients.py` and `logistic.py` next to `gradients.py` and `linear.py`. In the next lesson, *One-hot vectors for classes*, you write a class as a vector, the first step towards more than two classes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_step, result_fit, result_gradients):
    L.wrap_up(choice_step, result_gradients, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
