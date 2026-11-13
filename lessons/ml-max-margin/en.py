# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-max-margin. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-max-margin", "en")
    C = L.checks   # the Chinstraps and Gentoos of ml-knn, their street for every C, margin and slacks, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) and [Regularization and weight decay](/course/open/ml-regularization?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_boundary():
    quick_boundary = L.choice(
        "quick-boundary",
        question="A model with two features predicts class 1 where $w_1 x_1 + w_2 x_2 + b$ is above 0, and class 0 where it is below. What do the points where it is exactly 0 form?",
        options={
            "right": "A straight line: the decision boundary",
            "mistakes": "The training examples the model gets wrong",
            "curve": "A curve that bends around the examples",
        },
        correct="right",
        why={
            "mistakes": "Where $w_1 x_1 + w_2 x_2 + b = 0$ depends only on $w$ and $b$, not on the examples: it is a line, on whose two sides the model predicts different classes. See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "curve": "$w_1 x_1 + w_2 x_2 + b = 0$ has no squares or products of features: with two features, it is a straight line. See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "right": "A straight line, with class 1 on one side and class 0 on the other. Today you choose which straight line: the one with the most room on both sides.",
        },
    )
    quick_boundary
    return (quick_boundary,)


@app.cell(hide_code=True)
def quick_check_lambda():
    quick_lambda = L.choice(
        "quick-lambda",
        question="L2 regularization adds $\\lambda$ times the sum of the squared weights to the loss. What does a larger $\\lambda$ do?",
        options={
            "train": "It fits the training set better",
            "right": "It keeps the weights smaller",
            "rate": "It makes the steps of gradient descent larger",
        },
        correct="right",
        why={
            "train": "The penalty only makes the fit to the training set worse: it is the price of large weights. See *The idea* in [Regularization and weight decay](/course/open/ml-regularization?lang=en).",
            "rate": "$\\lambda$ is not the learning rate: it sets how heavily large weights count in the loss. See *The idea* in [Regularization and weight decay](/course/open/ml-regularization?lang=en).",
            "right": "Large weights cost more, so they stay smaller. Today the same sum of squared weights appears again: the smaller it is, the wider the street.",
        },
    )
    quick_lambda
    return (quick_lambda,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    By bill length and body mass, a straight line can separate the 44 Chinstraps from the 66 Gentoos of the training set without a single mistake. Then there are countless such lines: turn one a little, or move it a little, and it still separates them. Logistic regression stops at one of them, wherever gradient descent ends. Which one should you take for penguins you have not seen yet? The one with the most room on both sides, so that a new penguin that lies a little beyond the others of its species still lands on the right side. And when a few penguins pinch that room, it can be worth letting them in.

    **An everyday example.** On a narrow road you drive in the middle, as far as you can from both edges, so that a small swerve still keeps you on the road.
    """)
    return


@app.cell(hide_code=True)
def predict_support():
    guess_support = L.guess(
        "support",
        question="Of all the straight lines between the 44 Chinstraps and the 66 Gentoos of the training set, take the one with the widest street around it, without a penguin on the street. How many of the 110 penguins decide where that street lies?",
        options={
            "all": "All 110: each one pulls at the line",
            "half": "About half: those close to the other species",
            "right": "3",
        },
        reveal="**3**: a Chinstrap of 46 mm and 4,150 g, and two Gentoos, of 45.5 mm and 4,200 g and of 49.6 mm and 4,750 g. They lie on the edges of the street. Leave out any of the other 107, and the street stays exactly where it is.",
    )
    guess_support
    return (guess_support,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The margin.** A straight line $w \cdot x + b = 0$ splits the plane, as the decision boundary of logistic regression does. Give every Chinstrap $y = -1$ and every Gentoo $y = +1$. Then $y\,(w \cdot x + b)$ is positive for an example on its own side of the line, and negative on the wrong side; divided by the length of $w$, $\lVert w \rVert$, it is how far the example lies from the line. The **margin** of a line is the smallest of these distances over the training set: the room the nearest example has. When an example lies on the wrong side, the margin is negative.

        **The widest street.** The **maximum-margin classifier** takes, of all straight lines, the one with the largest margin. Think of a street between the two classes, as wide as it can be without an example on it: the line is its middle. The examples on its edges are the **support vectors**. Only they decide where the street lies: leave out any other example, and the street stays the same.

        **Scaled to 1.** Multiply $w$ and $b$ by the same positive number and the line stays where it is. So scale them until $y\,(w \cdot x + b) = 1$ for the support vectors. The edges of the street are then $w \cdot x + b = -1$ and $+1$, the margin is $1 / \lVert w \rVert$, and the street is $2 / \lVert w \rVert$ wide. The widest street is the smallest $\lVert w \rVert$, with $y\,(w \cdot x + b) \geq 1$ for every example: the smallest $\tfrac{1}{2} \lVert w \rVert^2$, the L2 penalty of regularization.

        **The soft margin.** Here three penguins pinch the street to a width of 0.11, in standardized units, and one Gentoo of the validation set ends up on the Chinstraps' side. A **soft margin** lets examples onto the street, or even across the line, at a price. The **slack variable** $\xi_i$, or slack, of example $i$ says how far it falls short of its edge of the street, counted in margins, so that 1 is all the way to the line: $\xi_i = 1 - y_i\,(w \cdot x_i + b)$, or 0 when that is below 0. It is 0 outside the street on its own side, between 0 and 1 on the street, and above 1 on the wrong side of the line. The soft margin takes the $w$ and $b$ with the smallest
        """),
        L.formula(r"\tfrac{1}{2} \lVert [[param:w]] \rVert^2 + [[hyper:C]] \sum_{i} [[loss:\xi_i]]"),
        mo.md(r"""
        $C$ is a hyperparameter: the price of one unit of slack. A large $C$ makes every slack expensive, so the street is narrow, with few examples on it; with $C$ large enough, it is the street without any example on it. A small $C$ makes room cheap: a wide street with many examples on it. Divide everything by $C$, and it is the sum of the slacks plus an L2 penalty on $w$ with $\lambda = 1 / (2C)$: a small $C$ is strong regularization.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **margin** | | the smallest distance from the training examples to the line, each on its own side; negative when one lies on the wrong side | `margin(w, b, X, y)` |
        | **support vector** | | a training example on an edge of the street or on it: only these decide where the street lies | |
        | **maximum-margin classifier** | | the straight line with the largest margin: the middle of the widest street | |
        | **soft margin** | | a street that lets examples onto it, or across the line, at a price per unit of slack | |
        | **slack variable**, $\xi_i$ | xi i | how far example $i$ falls short of its edge of the street, counted in margins: 0 outside the street on its own side, 1 on the line | `slacks(w, b, X, y)` |
        | $C$ | C | the price of one unit of slack: a hyperparameter | `C` |
        | `m[m > 2] = 2` | | assignment through a mask: every number of `m` where the mask is True becomes 2, the others stay | `m[m > 2] = 2` |
        """),
        L.watch_out("$C$ works the other way round from $\\lambda$: a larger $C$ means a narrower street and less regularization. And as for k-NN, the margin is a distance: standardize the features first, or body mass in grams decides almost alone."),
        mo.md(r"""
        The picture: the Chinstraps (triangles) and the Gentoos (squares) by bill length and body mass, the <span class="c-pred">line</span> of the widest street without a penguin on it, and dashed, the edges of the street. Left the training penguins, with a ring around the three support vectors; right the validation penguins, with a <span class="c-loss">purple ring</span> around the one on the wrong side.
        """),
    ])
    return


@app.cell(hide_code=True)
def figure_street():
    mo.Html(C.street_picture(C.HARD, {"train": "Training set", "val": "Validation set", "x": "bill length (mm)", "y": "body mass (g)",
                                       "alt": "Two charts of the Chinstraps and Gentoos by bill length and body mass, with a straight blue line that runs from the bottom left to the top right between the light Chinstraps and the heavy Gentoos, and close beside it two dashed lines: a narrow street. Left, the training set: three penguins are ringed on the edges of the street, a Chinstrap of 46 mm and 4,150 g and Gentoos of 45.5 mm and 4,200 g and of 49.6 mm and 4,750 g. Right, the validation set: one Gentoo, of 48.7 mm and 4,450 g, is ringed on the Chinstraps' side."}))
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Move the slider: it sets $C$, from 0.01 to 1,000, each stop about three times the one before, and the last ten times. The picture shows the street of the soft margin for that $C$, as the course finds it with a solver made for this loss: left the training penguins, with a ring around each one on an edge of the street or on it; right the validation penguins, with a purple ring around each one on the wrong side.
    """)
    return


@app.cell(hide_code=True)
def explore_c_cell():
    def _frame(_c):
        _f = C.facts(_c)
        _shown = C.c_text(_c, L.lang)
        _alt = (f"The street of the soft margin with C = {_shown}, {L.number(_f['width'], 2)} wide in standardized units. Left, the training set: "
                f"{_f['edge']} penguins ringed on the street or its edges, {_f['wrong']} of them on the wrong side of the line. Right, the validation set: "
                f"{_f['val_wrong']} of {_f['n_val']} penguins on the wrong side.")
        _said = (f"With $C$ = {_shown}: a street <span class=\"ea-num\">{L.number(_f['width'], 2)}</span> wide, in standardized units. "
                 f"{_f['edge']} training penguins lie on it or on its edges, {_f['wrong']} of them on the wrong side of the line. "
                 f"On the validation set: <span class=\"c-loss ea-num\">{_f['val_wrong']}</span> of {_f['n_val']} wrong.")
        return C.street_picture(_c, {"train": "Training set", "val": "Validation set", "x": "bill length (mm)",
                                      "y": "body mass (g)", "alt": _alt}), _said

    explore_c = L.explore("c", "C", _frame, start=1, values=C.CS, role="hyper",
                          task="Find the values of $C$ with no mistake on the validation set. How wide is the street there, against $C$ = 1,000?")
    explore_c
    return (explore_c,)


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    With $C$ = 0.1, 0.3 and 3, no validation penguin is wrong. Of these, $C$ = 0.1 gives the widest street: 1.34 wide, twelve times the 0.11 of $C$ = 1,000. 38 training penguins lie on it or on its edges, but none on the wrong side of the line. With $C$ = 1,000 the street is the one without a penguin on it, pinched by three penguins, and the Gentoo of 48.7 mm and 4,450 g from the validation set lands on the Chinstraps' side. A very small $C$, 0.01, makes the street so wide that 88 training penguins lie on it, 3 across the line, and 2 validation penguins are wrong. With only 39 validation penguins, one mistake more or less says little; but the wider street did not cost a single one.

    ### Recipe

    To find a line with room on both sides:

    1. Standardize the features, and give one class $y = -1$ and the other $y = +1$.
    2. For a line $w$, $b$: how far each example lies on its own side is `y * (X @ w + b)`; the margin is the smallest of these, divided by `np.linalg.norm(w)`.
    3. The slacks: `1 - y * (X @ w + b)`, with every value below 0 set to 0.
    4. The soft margin takes the $w$ and $b$ with the smallest $\tfrac{1}{2} \lVert w \rVert^2 + C \cdot$ (the sum of the slacks). Try a few values of $C$, and choose on the validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the k-NN lesson, split and standardized in the same way. The cell keeps the Chinstraps and the Gentoos, takes the $w$ and $b$ of the widest street without a penguin on it, as the course found them, and looks at how far each penguin lies on its own side. `(... == "Gentoo") * 2 - 1` gives $+1$ for a Gentoo and $-1$ for a Chinstrap: True counts as 1 and False as 0. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_street():
    import numpy as np
    import pandas as pd

    # the penguins of the k-NN lesson, split and standardized the same way
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    _train, _val = _order[:206], _order[206:274]
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
    _X_train, _y_train = (_X[_train] - _mean) / _std, _y[_train]
    _X_val, _y_val = (_X[_val] - _mean) / _std, _y[_val]

    # 1. only the Chinstraps and the Gentoos: y = -1 for a Chinstrap, +1 for a Gentoo
    _keep, _keep_val = _y_train != "Adelie", _y_val != "Adelie"
    _X_cg, _X_cg_val = _X_train[_keep], _X_val[_keep_val]
    _t = (_y_train[_keep] == "Gentoo") * 2 - 1
    _t_val = (_y_val[_keep_val] == "Gentoo") * 2 - 1
    print(len(_t), len(_t_val))

    # 2. the widest street without a penguin on it: how far each penguin lies on its own side
    _w, _b = np.array([-12.307, 13.335]), 3.984
    _sides = _t * (_X_cg @ _w + _b)
    print(np.min(_sides), np.sum(_sides < 1.01))
    print(_X[_train][_keep][_sides < 1.01])
    print(1 / np.linalg.norm(_w), 2 / np.linalg.norm(_w))

    # the same street on the validation set: who is on the wrong side?
    _sides_val = _t_val * (_X_cg_val @ _w + _b)
    print(np.sum(_sides_val < 0), _X[_val][_keep_val][_sides_val < 0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    110 training penguins and 39 validation penguins. The smallest value of `_sides` is 0.9998: 1, as the scaling promises, up to the rounding of $w$; and exactly 3 penguins have less than 1.01: the support vectors, a Gentoo of 49.6 mm and 4,750 g, a Chinstrap of 46 mm and 4,150 g and a Gentoo of 45.5 mm and 4,200 g. The margin is $1 / \lVert w \rVert$ = 0.055, the street 0.110 wide. On the validation set, one penguin lies on the wrong side: the Gentoo of 48.7 mm and 4,450 g, lighter than the line expects for its bill.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_margin():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The margin of a line

        `margin(w, b, X, y)` should give back the margin of the line $w \cdot x + b = 0$ on the examples `X` with labels `y`, $-1$ or $+1$, as step 2 of the recipe. Replace the two `...`. The check calls `margin(np.array([1.0, 1.0]), 0.0, X, y)` with ten made-up penguins, already standardized: five Chinstraps and five Gentoos. It also tries the same line with $w$ and $b$ three times as large, two other lines and three measurements.
        """),
        L.task("Make `margin` give back the distance from the line to the nearest example, negative when an example lies on the wrong side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_margin():
    box_margin = L.code_box("margin", lines=6)
    box_margin
    return (box_margin,)


@app.cell(hide_code=True)
def check_margin(box_margin):
    result_margin = L.check("margin", box_margin.value)
    result_margin
    return (result_margin,)


@app.cell(hide_code=True)
def hints_margin():
    L.hints(
        "margin",
        [
            "Step 2 of the recipe, as in the worked example: first how far each example lies on its own side, then the smallest of those.",
            "For an example on its own side, `y` times its score is positive. Which NumPy function gives the smallest of those? And what turns it into a distance, as in *Scaled to 1*?",
            "`sides = y * (X @ w + b)` and `return np.___(sides) / np.linalg.___(w)`.",
        ],
        why="For the ten made-up penguins, the smallest value in `sides` is 1.1, for two Chinstraps and a Gentoo; the length of `[1.0, 1.0]` is 1.414, so the margin is 1.1 / 1.414 = 0.778.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_slacks():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The slacks

        `slacks(w, b, X, y)` should give back the slack of every example: how far it falls short of its edge of the street, counted in margins, and 0 outside the street on its own side, as step 3 of the recipe. Replace the two `...`: the first line works out $1 - y_i\,(w \cdot x_i + b)$ for every example, the second makes the mask of the values below 0, and the line after it sets those to 0. The check calls `slacks(np.array([0.5, 0.5]), 0.0, X, y)` with the same ten penguins, and tries three other lines and three measurements.
        """),
        L.task("Make `slacks` give back one slack for every example."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slacks():
    box_slacks = L.code_box("slacks", lines=8)
    box_slacks
    return (box_slacks,)


@app.cell(hide_code=True)
def check_slacks(box_slacks):
    result_slacks = L.check("slacks", box_slacks.value)
    result_slacks
    return (result_slacks,)


@app.cell(hide_code=True)
def hints_slacks():
    L.hints(
        "slacks",
        [
            "Step 3 of the recipe. An example exactly on its edge has `y * (X @ w + b)` equal to 1, and a slack of 0.",
            "Work out the formula for all examples at once, as in the margin: one number for each. Then a mask picks the values below 0, and one assignment through that mask sets them all to 0, as in *New words and symbols*.",
            "`s = 1 - y * (___)` and `to_zero = s ___ 0`.",
        ],
        why="With `np.array([0.5, 0.5])`, the third Chinstrap has $y\\,(w \\cdot x + b)$ = 0.7: it lies on the street, 0.3 margins short of its edge. The first lies exactly on its edge, at 1.0, and the fourth Gentoo beyond it, at 1.1: both get 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_support():
    mo.md(r"""
    ### Exercise 3. Leave one out
    """)
    return


@app.cell(hide_code=True)
def choice_support_cell():
    choice_support = L.choice(
        "not-a-support-vector",
        question="The widest street without a penguin on it has three support vectors. You leave out one of the other 107 training penguins, and look for the widest street again. What happens?",
        options={
            "moves": "It moves a little: every penguin pulls at the line",
            "right": "It stays exactly where it is",
            "narrower": "It gets narrower: there is one penguin less to hold it open",
        },
        correct="right",
        why={
            "moves": "That is how logistic regression works: every example adds to its loss. The widest street only feels the penguins on its edges. See *The widest street* in *The idea*.",
            "narrower": "The penguins away from the street do not hold it open; the street was as wide as the three support vectors allowed. Without one of the others, it is still the widest. See *The widest street* in *The idea*.",
            "right": "The three support vectors still pinch it in the same place, and no other penguin was in the way: the street stays. You could drop the other 107 and find the same street from the three alone.",
        },
    )
    choice_support
    return (choice_support,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The line is the decision boundary of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en); but where logistic regression lets every example add to its loss, the widest street only feels its support vectors. The distance to the line uses the length of a vector from [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), and $\tfrac{1}{2} \lVert w \rVert^2$ is the L2 penalty of [Regularization and weight decay](/course/open/ml-regularization?lang=en), with $C$ in the role of $1 / (2\lambda)$. In the next lesson, *Many classes from two: one-vs-rest and one-vs-one*, you turn a classifier for two classes, such as this one, into one for all three species.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_support, result_margin, result_slacks):
    L.wrap_up(result_margin, result_slacks, choice_support)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
