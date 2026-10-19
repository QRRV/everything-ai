# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-digits-linear. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow multinomial logistic regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-digits-linear", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en), [Broadcasting](/course/open/np-broadcasting?lang=en) and [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`diff` has shape (6, 3) and `X` has shape (6, 4). Which product has shape (3, 4)?",
        options={
            "x-t": "`X.T @ diff`",
            "right": "`diff.T @ X`",
            "no-t": "`diff @ X`",
        },
        correct="right",
        why={
            "x-t": "`X.T @ diff` is (4, 6) times (6, 3), so it has shape (4, 3). Which of the two has to turn, so that the 3 comes first? See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "no-t": "`diff @ X` puts (6, 3) before (6, 4): the inner numbers, 3 and 6, differ, so Python stops. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "right": "`diff.T` has shape (3, 6), and (3, 6) times (6, 4) gives (3, 4).",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def quick_check_keepdims():
    quick_keepdims = L.choice(
        "quick-keepdims",
        question="`m` has shape (4, 3). Which line divides each row of `m` by the sum of that row?",
        options={
            "total": "`m / m.sum()`",
            "columns": "`m / m.sum(axis=0)`",
            "right": "`m / m.sum(axis=1, keepdims=True)`",
        },
        correct="right",
        why={
            "total": "That divides every number by the sum of the whole matrix: then all twelve together add up to 1, not each row. See [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "columns": "`m.sum(axis=0)` has shape (3,): one sum per column, so each column is divided by its own sum. Which axis adds up along a row? See [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "right": "`m.sum(axis=1, keepdims=True)` has shape (4, 1): one sum per row, as a column. Broadcasting then divides each row by its own sum.",
        },
    )
    quick_keepdims
    return (quick_keepdims,)


@app.cell(hide_code=True)
def quick_check_slopes():
    quick_slopes = L.choice(
        "quick-slopes",
        question="A model for the digits 0, 1 and 2 gives an image of a 2 the probabilities 0.1, 0.3 and 0.6. What are the slopes of its cross-entropy for the three logits?",
        options={
            "right": "$(0.1,\\ 0.3,\\ -0.4)$",
            "first": "$(-0.9,\\ 0.3,\\ 0.6)$",
            "plain": "$(0.1,\\ 0.3,\\ 0.6)$",
        },
        correct="right",
        why={
            "first": "That takes the 1 off the class 0. Which class is right for an image of a 2? See the recipe in [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en).",
            "plain": "Those are the probabilities themselves. What do you take off them? See the recipe in [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en).",
            "right": "The probabilities minus the one-hot label $(0, 0, 1)$: only the right class gets a negative slope, so gradient descent raises its logit.",
        },
    )
    quick_slopes
    return (quick_slopes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Training logistic regression](/course/open/ml-train-logistic?lang=en) your project trained a classifier for two classes. Handwritten digits have ten: 0 to 9. You know all the parts for more classes: softmax turns logits into probabilities, cross-entropy says how wrong they are, and you know its slopes. With the same loop, they make a classifier for ten classes. Today you train it on real images, 6,000 handwritten digits, and judge it the way you learned: by its accuracy next to a baseline, and by which digits it mixes up. That is the third step of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           2, here_label="← today", title="Where your project stands: your own neural network"),
        mo.md(r"""
        **An everyday example.** Sorting machines at the post office read the postcode that someone wrote by hand on an envelope, one digit at a time, and send the letter on to the right place.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share-right",
        question="A model with one weighted sum of the 784 pixels per digit trains on 6,000 handwritten digits. How many of 1,000 new digits do you think it reads right?",
        options={
            "hundred": "About 100: hardly more than guessing",
            "half": "About 500",
            "most": "About 900",
            "all": "All but a few",
        },
        reveal="**About 900.** In the demo at the end of this lesson, your own model reads 890 of the 1,000 new digits right. Always saying the most common digit, a 1, gets 126 right: that is the baseline. But one weighted sum per digit cannot read them all: handwriting differs too much, and in the demo you see which digits it mixes up.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An image of 28 by 28 pixels is 784 numbers. Read row after row, they make one row of $\mathbf{X}$: an example with 784 features, one per pixel, from 0 for paper to 1 for ink. **Softmax regression** is logistic regression for more than two classes. Every class $k$ gets its own weights $\mathbf{w}_k$ and its own bias term $b_k$, so its own logit for example $i$:
        """),
        L.formula(r"z_{ik} = [[param:\mathbf{w}_k]] \cdot \mathbf{x}_i + [[param:b_k]] \qquad k = 0, 1, \dots, K - 1"),
        mo.md(r"""
        Softmax turns the $K$ logits of an example into $K$ probabilities $\hat{y}_{ik}$, as in [Many classes: softmax](/course/open/ml-softmax?lang=en). The weights of class $k$ are row $k$ of the **weight matrix** $\mathbf{W}$, of shape $(K, d)$: here 10 rows of 784. For all examples and classes at once, the logits are `X @ W.T + b`, of shape $(n, K)$: one row per example, one column per class.

        The slopes come from [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en): for one example, the slope for logit $k$ is $\hat{y}_k - y_k$, where $y_k$ is 1 for the right class and 0 for the others. The chain rule multiplies it by the feature, as for log loss, and the cross-entropy of $n$ examples is their mean:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_{kj}}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_{ik}]] - y_{ik}) \, x_{ij} \qquad [[grad:\frac{\partial \mathcal{L}}{\partial b_k}]] = \frac{1}{n} \sum_{i=1}^{n} ([[pred:\hat{y}_{ik}]] - y_{ik})"),
        mo.md(r"""
        In code, `diff = P - np.eye(K)[y]` holds the differences $\hat{y}_{ik} - y_{ik}$, one row per example. Row $k$ of `diff.T` holds those of class $k$ for all examples; its dot product with column $j$ of `X` is the sum over $i$ in the formula. So `diff.T @ X / n` gives every slope at once, with the shape of `W`, $(K, d)$. The slopes for `b` are the mean of each column of `diff`. Side by side with your logistic regression:

        | | Logistic regression | Softmax regression |
        |---|---|---|
        | weights | `w`, shape $(d,)$ | `W`, shape $(K, d)$: one row per class |
        | bias term | `b`, one number | `b`, shape $(K,)$: one per class |
        | probabilities | `1 / (1 + np.exp(-(X @ w + b)))` | softmax along each row of `X @ W.T + b` |
        | differences | `p - y` | `P - np.eye(K)[y]` |
        | slopes for the weights | `(p - y) @ X / n` | `diff.T @ X / n` |
        | slopes for the bias term | `(p - y).mean()` | `diff.mean(axis=0)` |
        | one step | `w = w - lr * grad_w` | `W = W - lr * grad_W` |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **softmax regression** | | logistic regression for more than two classes: one weighted sum per class, turned into probabilities with softmax; statistics calls it multinomial logistic regression | `X @ W.T + b`, then softmax |
        | $\mathbf{W}$ | capital W | the **weight matrix**: one row of weights per class, shape $(K, d)$ | `W` |
        | $w_{kj}$ | w k j | the weight of class $k$ for feature $j$ | `W[k, j]` |
        | **`np.load(path)`** | | reads arrays that NumPy saved in a file; an `.npz` file holds several, each under its own name | `np.load(path)["train_images"]` |
        | **`.reshape(rows, columns)`** | reshape | the same numbers in another shape, filled row after row | `X[0].reshape(28, 28)` |
        """),
        L.watch_out("Softmax goes along each row: one example, all its classes. `Z.max(axis=1)` has shape $(n,)$ and does not fit `Z`, of shape $(n, K)$. With `keepdims=True` it has shape $(n, 1)$, a column, and every row has its own largest logit taken off. The same holds for the sum you divide by."),
        mo.Html(C.images_picture(C.examples(), [str(_k) for _k in range(10)], "The first image of each digit in the training set",
                                 "Ten handwritten digits, 0 to 9, each drawn from its 784 numbers as 28 rows of 28 small squares: the more ink, the stronger the square.")),
        mo.md(r"""
        Each image is one row of $\mathbf{X}$, drawn back as 28 rows of 28 pixels: the more ink, the stronger the square. The same digit looks different every time someone writes it. One weighted sum per digit has to cope with all of that.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One step of gradient descent for softmax regression, on $n$ examples and $K$ classes:

    1. Give every example its $K$ logits: `Z = X @ W.T + b`, shape $(n, K)$.
    2. Softmax along each row: `E = np.exp(Z - Z.max(axis=1, keepdims=True))` and `P = E / E.sum(axis=1, keepdims=True)`.
    3. Take the one-hot labels off: `diff = P - np.eye(K)[y]`.
    4. Work out every slope: `grad_W = diff.T @ X / n` and `grad_b = diff.mean(axis=0)`.
    5. Step against the slopes, `W = W - lr * grad_W` and `b = b - lr * grad_b`, and repeat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The digits come from QMNIST, a set of handwritten digits with an open license, which Chhavi Yadav and Léon Bottou rebuilt in 2019 from the same forms as the well-known MNIST. The course has its first 6,000 training images and its first 1,000 test images in `data/qmnist-small.npz`, each image as 784 numbers from 0 to 255; divided by 255, they go from 0 to 1. The first cell loads the training images and shows the first one. The second takes one step from $\mathbf{W} = 0$ and $\mathbf{b} = 0$, with $\eta$ = 1, and prints the cross-entropy before and after. The comments with `#` say which step of the recipe each part is. Run the two cells.
    """)
    return


@app.cell
def example_data():
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load("data/qmnist-small.npz")
    X = data["train_images"] / 255
    y = data["train_labels"]
    print(X.shape)
    print(y[:10])
    # the first image: its 784 numbers as 28 rows of 28
    plt.imshow(X[0].reshape(28, 28), cmap="gray")
    plt.show()
    return X, data, np, plt, y


@app.cell
def example_step(X, np, y):
    _W = np.zeros((10, 784))
    _b = np.zeros(10)

    def _cross_entropy(W, b):
        _Z = X @ W.T + b
        _E = np.exp(_Z - _Z.max(axis=1, keepdims=True))
        _P = _E / _E.sum(axis=1, keepdims=True)
        return -np.mean((np.eye(10)[y] * np.log(_P)).sum(axis=1))

    # 1. ten logits for every image
    _Z = X @ _W.T + _b
    print(_Z.shape)
    # 2. softmax along each row
    _E = np.exp(_Z - _Z.max(axis=1, keepdims=True))
    _P = _E / _E.sum(axis=1, keepdims=True)
    print(_P[0])
    # 3. the probabilities minus the one-hot labels
    _diff = _P - np.eye(10)[y]
    print(_diff[0])
    # 4. every slope at once
    _grad_W = _diff.T @ X / len(y)
    _grad_b = _diff.mean(axis=0)
    print(_grad_W.shape)
    print(_grad_b)
    # 5. one step, and the cross-entropy before and after
    _lr = 1.0
    print(_cross_entropy(_W, _b), _cross_entropy(_W - _lr * _grad_W, _b - _lr * _grad_b))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `X` has shape (6000, 784): 6,000 images of 784 pixels. The first ten labels are 5, 0, 4, 1, 9, 2, 1, 3, 1 and 4, and the first image is indeed a 5, in white: with `cmap="gray"`, 0 is black and 1 white. `Z` has shape (6000, 10), ten logits per image. With all weights 0, every logit is 0 and every probability 0.1: the model does not know anything yet. For that 5, the differences are 0.1 for the nine other digits and −0.9 for the 5. The slopes for `W` have the shape of `W`, (10, 784). Each slope for `b` is 0.1 minus the share of that digit in the training set, so close to 0: about −0.012 for the 1s, the most common digit. One step brings the cross-entropy down from 2.30, the loss of a probability of 0.1 for the right digit, to 1.47. In exercise 3 your `fit` repeats this step in a loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many numbers?
    """)
    return


@app.cell(hide_code=True)
def choice_count_cell():
    choice_count = L.choice(
        "how-many",
        question="How many numbers does softmax regression learn for images of 784 pixels and 10 digits?",
        options={
            "one-row": "794",
            "no-b": "7,840",
            "right": "7,850",
        },
        correct="right",
        why={
            "one-row": "That is one weight per pixel, plus one bias term per digit. How many weights does each digit get? See *The idea*.",
            "no-b": "That is all the weights: 10 rows of 784. What else does each digit get? See *The idea*.",
            "right": "10 rows of 784 weights, 7,840, plus 10 bias terms: 7,850. Logistic regression on the same pixels learns 785.",
        },
    )
    choice_count
    return (choice_count,)


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The slopes of the cross-entropy

        Finish `cross_entropy_gradients(X, y, W, b)`. It gets the examples `X`, the labels `y` as class numbers, the weights `W`, one row per class, and the bias terms `b`, one per class. It gives back the slopes of the cross-entropy: `grad_W`, with the shape of `W`, and `grad_b`, with the shape of `b`. Replace the `...` with steps 1 to 4 of the recipe; `len(b)` is the number of classes. The check calls `cross_entropy_gradients([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], [[0, 0], [0, 0], [0, 0]], [0, 0, 0])`: four examples with two features, and three classes. That should give about `([[-0.0833, 0.1667], [0.1667, 0.1667], [-0.0833, -0.3333]], [0.0833, 0.0833, -0.1667])`. It also tries other examples, weights, bias terms and numbers of classes, also with very large logits. When your function is right, the check saves it in your project, as `work/projects/nn/softmax_gradients.py`.
        """),
        L.task("Make `cross_entropy_gradients` give back the slopes of the cross-entropy for `W` and for `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("cross-entropy-gradients", lines=14)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("cross-entropy-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "cross-entropy-gradients",
        [
            "Steps 1 to 4 of the recipe, in the same order: the logits, softmax along each row, the differences, then the two slopes.",
            "`W` has one row per class, so the logits are `X @ W.T + b`. Which shape must `grad_W` have, and which of `diff` and `X` do you turn to get it?",
            "`Z = X @ W.T + b`, then `E` and `P` as in step 2, `diff = P - np.eye(len(b))[y]`, `grad_W = ___ @ X / len(y)` and `grad_b = diff.mean(axis=___)`.",
        ],
        why="With `W` and `b` all 0, every probability is a third, so each row of `diff` is a third minus the one-hot label: −2/3 for the right class and 1/3 for the other two. `diff.T @ X` adds up, for each class, the differences of the examples that have each feature; divided by 4, that gives the slopes for `W`. The mean of each column of `diff` gives about `[0.0833, 0.0833, -0.1667]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Train

        `fit(X, y, K, lr, steps)` imports your `cross_entropy_gradients` from your project, starts with `W` and `b` all 0, `K` rows of weights and `K` bias terms for `K` classes, and runs its loop `steps` times: the same loop as in your `fit` for logistic regression. Replace the `...` with one update step: get both slopes with `softmax_gradients.cross_entropy_gradients(X, y, W, b)`, and move `W` and `b` against them. The check calls `fit([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], 3, 0.5, 100)`, with the examples of exercise 2. It also tries other data, numbers of classes, learning rates and numbers of steps. When your function is right, the check saves it in your project, as `work/projects/nn/softmax.py`.
        """),
        L.task("Make `fit` train `W` and `b` with your own `cross_entropy_gradients`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-softmax", lines=13)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit, result_gradients):
    result_gradients  # check again when exercise 2 saves your module
    result_fit = L.check("fit-softmax", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-softmax",
        [
            "Step 5 of the recipe, with the slopes from your own function: the same loop as in your `fit` for logistic regression.",
            "The function lives in the module `softmax_gradients`: call it through the module. It gives back two things at once: how do you give each its own name?",
            "`grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)`, then `W = W - ___` and `b = b - ___`.",
        ],
        why="`softmax_gradients.cross_entropy_gradients(X, y, W, b)` gives both slopes; `grad_W, grad_b = ...` catches them. `W = W - lr * grad_W` and `b = b - lr * grad_b` step against them. After 100 rounds on the example, class 2, whose two examples have the second feature, has a weight of about 3.61 for it. Class 1, whose example has neither feature, has negative weights for both: about −2.18 and −1.67.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below trains a model with your own `fit` on the 6,000 training digits, with $\eta$ = 1 and 100 steps, and measures it on the 1,000 test digits, which it never saw. That takes a few seconds. The cell uses your own module, so it works once exercise 3 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_digits(result_fit):
    result_fit  # run again when exercise 3 saves your module
    _model = C.trained()
    if _model is None:
        _out = mo.md("*Finish exercise 3 first: then your `fit` is in your project, and this cell uses it.*")
    elif _model == "slow":
        _out = mo.md("*Your `fit` in `work/projects/nn/softmax.py` takes more than a minute on the 6,000 digits. Does it loop over the examples one at a time? The recipe does every example at once, with `@`.*")
    elif isinstance(_model, str):
        _out = mo.md("*Your `fit` gets its slopes from `cross_entropy_gradients` in `work/projects/nn/softmax_gradients.py`, and that file is missing, or no longer gives the right slopes. Check exercise 2 again; if you changed that file, delete it first.*")
    elif not _model:
        _out = mo.md("*The `fit` in `work/projects/nn/softmax.py` no longer trains right, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*")
    else:
        _W, _b = _model
        _accuracy = C.accuracy(_W, _b)
        _digit, _baseline = C.baseline()
        _cm = C.confusion(_W, _b)
        _true, _said, _times = C.most_confused(_cm)
        _out = mo.vstack([
            mo.md(f"""Your model reads <span class="ea-num">{L.number(_accuracy * 1000, 0)}</span> of the 1,000 test digits right: an accuracy of <span class="ea-num">{L.number(_accuracy, 3)}</span>. Always saying the most common digit of the training set, a {_digit}, gets <span class="ea-num">{L.number(_baseline, 3)}</span>. Your model beats the baseline by far."""),
            mo.Html(C.confusion_picture(_cm, (_true, _said), {"title": "Your model on the 1,000 test digits", "label": "the confusion matrix",
                                                              "caption": "cm[label, pred]", "lines": ["rows: the true digit", "columns: the digit your model says",
                                                                                                         f"in bold: {_true} read as {_said}"]})),
            mo.md(f"""On the diagonal are the digits it reads right; every other box counts digits it took for another digit. It most often mistakes the digit {_true} for the digit {_said}: {_times} times. Some digits it reads almost always right, such as the 0s and the 1s; others less often, such as the 2s and the 8s."""),
            mo.Html(C.images_picture(_W, [str(_k) for _k in range(10)], "The ten rows of your W, as images",
                                     "Ten images of 28 by 28, one per digit: the weights of that digit, orange where ink raises its logit and slate gray where ink lowers it.",
                                     signed=True)),
            mo.md("""Each image is one row of your `W`: the 784 weights of one digit, back as 28 rows of 28. Ink on an orange pixel raises the logit of that digit; ink on a gray pixel lowers it. The 0 has an orange ring with gray in the middle: ink in the middle speaks against a 0. Each row is one fixed pattern per digit, and a digit written in another place or at a slant matches it less well. That is why a linear model stops at about 9 in 10. A neural network with a hidden layer, later in this project, learns more patterns per digit."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The gradient of softmax with cross-entropy](/course/open/ml-softmax-gradient?lang=en) you found the slopes $\hat{y}_k - y_k$ for one example; in [Training logistic regression](/course/open/ml-train-logistic?lang=en) you wrote the loop that steps against slopes. With a weight matrix of one row per class, they now read handwritten digits. [Always beat a simple baseline](/course/open/ml-baselines?lang=en) and [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) gave you the ways to judge it: its accuracy next to the baseline, and the confusion matrix. Your project `nn` now has `softmax_gradients.py` and `softmax.py` as well. In the next lesson, *The machine learning workflow*, you put these steps in order: from the question to the metric, the data, a baseline and a model, and round again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count, result_fit, result_gradients):
    L.wrap_up(choice_count, result_gradients, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
