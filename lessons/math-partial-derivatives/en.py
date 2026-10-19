# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-partial-derivatives. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-partial-derivatives", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Derivative rules](/course/open/math-derivative-rules?lang=en) and [Two parameters: the gradient](/course/open/math-gradient?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_product():
    quick_product = L.choice(
        "quick-product",
        question="What is the derivative of $x \\cdot e^x$?",
        options={
            "times": "$e^x$: the derivative of $x$, 1, times that of $e^x$",
            "right": "$e^x + x \\cdot e^x$: by the product rule",
            "power": "$x \\cdot e^{x-1}$: by the power rule",
        },
        correct="right",
        why={
            "times": "The derivative of a product is not the product of the derivatives. See [Derivative rules](/course/open/math-derivative-rules?lang=en).",
            "power": "$x$ is in the exponent of $e^x$, so the power rule does not apply there. See [Derivative rules](/course/open/math-derivative-rules?lang=en).",
            "right": "The product rule: $1 \\cdot e^x + x \\cdot e^x$.",
        },
    )
    quick_product
    return (quick_product,)


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="The gradient of a loss is $(2,\\ -5)$ at some point. Which way does the loss rise fastest?",
        options={
            "right": "Along $(2,\\ -5)$: the gradient points uphill",
            "down": "Along $(-2,\\ 5)$: the opposite way",
            "first": "Along the first parameter only: it comes first",
        },
        correct="right",
        why={
            "down": "That is where the loss falls fastest, the way gradient descent goes. See [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "first": "The gradient points in one direction, made of both partial derivatives at once. See [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "right": "The gradient points in the direction in which the loss rises fastest; its opposite is the way down.",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Two parameters: the gradient](/course/open/math-gradient?lang=en) you measured each partial derivative with a tiny step: one more run of the loss for every parameter. A neural network has millions of parameters, so that would be millions of runs for a single step of training. With the derivative rules, each partial derivative becomes a formula instead. And in a network, one value often reaches the loss along more than one path; the chain rule then needs one more idea.

    **An everyday example.** On a warmer day, an ice-cream stall sells more in two ways: more people walk past, and each of them buys more. One degree warmer raises the sales along both ways at once, and the two effects add up.
    """)
    return


@app.cell(hide_code=True)
def predict_partial():
    guess_partial = L.guess(
        "partial-x",
        question="$f(x, y) = x^2 \\cdot y$. What do you think $\\frac{\\partial f}{\\partial x}$ is at $x = 3$, $y = 2$?",
        options={
            "two_x": "6: the power rule gives $2x$",
            "right": "12: $2x$, times the $y$ in front",
            "x_sq": "9: the $x^2$ that multiplies $y$",
        },
        reveal="**12.** For the partial derivative for $x$, $y$ is held fixed: it is just a number in front, like the 3 in $3x^2$. So $\\frac{\\partial f}{\\partial x} = 2x \\cdot y$, and that is 12 at $x = 3$, $y = 2$. The 9 is $x^2$: that is the partial derivative for $y$, with $x$ held fixed.",
    )
    guess_partial
    return (guess_partial,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The partial derivative $\frac{\partial f}{\partial x}$ of a function of several variables is the derivative for $x$, with every other variable held fixed. Formally: treat the others as numbers, and use the derivative rules. For $f(x, y) = x^2 \cdot y + e^y$ that gives $\frac{\partial f}{\partial x} = 2x \cdot y$, because $e^y$ is a fixed number for $x$ and its derivative is 0; and $\frac{\partial f}{\partial y} = x^2 + e^y$. The gradient $\nabla f$ is the list of all partial derivatives, as before, but now as exact formulas.

        Often a value reaches the output along more than one path. Say $z = u \cdot v$, and both $u$ and $v$ depend on $t$. A small change in $t$ changes $z$ through $u$ and through $v$, and both effects count. The **multivariable chain rule**: follow every path from $t$ to $z$, multiply the local derivatives along each path, and add the paths.
        """),
        L.formula(r"[[grad:\frac{dz}{dt}]] = \frac{\partial z}{\partial u} \cdot \frac{du}{dt} + \frac{\partial z}{\partial v} \cdot \frac{dv}{dt}"),
        mo.md(r"""
        The product rule is one case of it: for $z = u \cdot v$, $\frac{\partial z}{\partial u} = v$ and $\frac{\partial z}{\partial v} = u$, so $\frac{dz}{dt} = v \cdot u' + u \cdot v'$. In a neural network, one value often feeds several others; the derivative of the loss for that value is then a sum, with one term for each path.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **local derivative** | | the derivative of a value for one of its own inputs, from its own formula alone, such as $\frac{du}{dt} = 2t$ for $u = t^2$ | `du_dt` |
        | **multivariable chain rule** | | when a value reaches the output along several paths: multiply the local derivatives along each path, and add the paths | `dz_du * du_dt + dz_dv * dv_dt` |
        | $\frac{\partial z}{\partial u}$ | partial z by partial u | the derivative of $z$ for $u$, with the other inputs of $z$ held fixed | `dz_du` |
        | $\frac{dz}{dt}$ | d z by d t | the derivative of $z$ for $t$, along every path | `dz_dt` |
        """),
        L.watch_out("Along one path, multiply; across paths, add. Leaving out a path misses part of the change, and multiplying the paths together gives nonsense."),
        mo.md(r"""
        The picture: $z = u \cdot v$ with $u = t^2$ and $v = 3t + 1$, at $t = 2$. Each arrow carries its <span class="c-grad">local derivative</span>; below $t$, $u$ and $v$, the derivative of $z$ for each.
        """),
        C.paths_picture({"title": "Two paths from t to z", "t": "4 · 7 + 3 · 4 = 40", "u": "∂z/∂u = 7", "v": "∂z/∂v = 4",
                         "z": "",
                         "alt": "Four boxes: t, with the value 2, on the left; u = t², with 4, and v = 3t + 1, with 7, in the middle; z = u · v, with 28, on the right. Arrows from t to u and to v, and from u and v to z, each with its local derivative: du/dt = 4, dv/dt = 3, ∂z/∂u = 7 and ∂z/∂v = 4. Below t: 4 · 7 + 3 · 4 = 40."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the derivatives of a formula with several variables:

    1. For a partial derivative, hold every other variable fixed: treat it as a number, and use the derivative rules.
    2. Do that for every variable, in order: together they are the gradient.
    3. When a value reaches the output along several paths, multiply the local derivatives along each path.
    4. Add the paths.
    5. Check with a tiny step: change one variable at a time.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The gradient of $x^2 \cdot y + e^y$ at $x = 3$, $y = 2$; then $z = u \cdot v$ with $u = t^2$ and $v = 3t + 1$, at $t = 2$, as in the picture. The comments with `#` say which step of the recipe each part is. The path through $v$ is not finished: run the cell as it is, then replace the `0` where the comment says, and run it again.
    """)
    return


@app.cell
def example_partials():
    import numpy as np

    def _f(x, y):
        return x ** 2 * y + np.exp(y)

    _x = 3
    _y = 2
    # 1. for x, y is a number: (x² · y)' = 2x · y, and eʸ gives 0
    _df_dx = 2 * _x * _y
    # 1. for y, x is a number: (x² · y)' = x², and (eʸ)' = eʸ
    _df_dy = _x ** 2 + np.exp(_y)
    # 2. together, in order: the gradient
    print(_df_dx, _df_dy)
    # 5. check: one variable at a time, with a tiny step
    _h = 0.001
    print((_f(_x + _h, _y) - _f(_x, _y)) / _h, (_f(_x, _y + _h) - _f(_x, _y)) / _h)

    # z = u · v, with u = t² and v = 3t + 1, at t = 2
    _t = 2
    _u = _t ** 2
    _v = 3 * _t + 1
    # 3. along each path, multiply: ∂z/∂u = v, times du/dt = 2t; ∂z/∂v = u, times dv/dt = 3
    _path_u = _v * (2 * _t)
    _path_v = 0   # the path through v: replace the 0 by ∂z/∂v times dv/dt
    # 4. add the paths
    print(_path_u, _path_v, _path_u + _path_v)
    return


@app.cell(hide_code=True)
def text_after_example():
    _e2 = 7.38905609893065
    mo.md(f"""
    The gradient is $(12,\\ {L.number(9 + _e2, 3).replace(",", "{,}")})$; the tiny steps give about the same, a little off. For $z$, the path through $u$ gives 28; with the 0 still in place, the sum is 28 too, which is too little. The path through $v$ gives 12: together 40. Check it by writing $z$ out: $z = t^2 \\cdot (3t + 1) = 3t^3 + t^2$, with the derivative $9t^2 + 2t$, which is 40 at $t = 2$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_partial():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Hold x fixed
    """)
    return


@app.cell(hide_code=True)
def choice_partial_cell():
    choice_partial = L.choice(
        "partial-y",
        question="$f(x, y) = x \\cdot y + y^2$. What is $\\frac{\\partial f}{\\partial y}$ at $x = 2$, $y = 3$?",
        options={
            "square": "6: the derivative of $y^2$",
            "value": "15: the value of $f$ there",
            "right": "8: $x + 2y$, from both parts",
        },
        correct="right",
        why={
            "square": "That is the second part only. The first part, $x \\cdot y$, has $y$ in it too: with $x$ held fixed, its derivative for $y$ is $x$.",
            "value": "$2 \\cdot 3 + 3^2 = 15$ is the value of $f$, not its slope.",
            "right": "For $y$, $x$ is a number: $(x \\cdot y)' = x = 2$ and $(y^2)' = 2y = 6$. Together 8.",
        },
    )
    choice_partial
    return (choice_partial,)


@app.cell(hide_code=True)
def text_exercise_gradient():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The gradient as formulas

        `grad_f(x, y)` should give the gradient of `f(x, y)`, which is $x \cdot y^2 + \log(x)$, as a list: first the partial derivative for `x`, then for `y`. Write the body of the function, by steps 1 and 2 of the recipe. The check calls `grad_f(2, 3)`, which should give `[9.5, 12]`, and tries other points, with `x` above 0.
        """),
        L.task("Make `grad_f` give the two exact partial derivatives of `f`, in a list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradient():
    box_gradient = L.code_box("gradient-formula", lines=11)
    box_gradient
    return (box_gradient,)


@app.cell(hide_code=True)
def check_gradient(box_gradient):
    result_gradient = L.check("gradient-formula", box_gradient.value)
    result_gradient
    return (result_gradient,)


@app.cell(hide_code=True)
def hints_gradient():
    L.hints(
        "gradient-formula",
        [
            "Step 1 of the recipe, once for `x` and once for `y`: which variable is a fixed number each time?",
            "For `x`, `y ** 2` is a number in front of `x`, and `np.log(x)` has its own rule. For `y`, `x` is a number in front of `y ** 2`, and `np.log(x)` is a fixed number.",
        ],
        why="At $x = 2$, $y = 3$: for $x$, $y^2 + \\frac{1}{x} = 9 + 0.5 = 9.5$; for $y$, $2xy = 12$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_paths():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Two paths

        `dz_dt(t)` should give the derivative of $z = u \cdot v$ for $t$, with $u = t^2$ and $v = e^{2t}$. Write the body of the function, by steps 3 and 4 of the recipe: the local derivatives along each path, and the two paths added. The check calls `dz_dt(1)`, which should give about `29.556`, and tries other values of `t`.
        """),
        L.task("Make `dz_dt` add the two paths from `t` to `z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paths():
    box_paths = L.code_box("two-paths", lines=11)
    box_paths
    return (box_paths,)


@app.cell(hide_code=True)
def check_paths(box_paths):
    result_paths = L.check("two-paths", box_paths.value)
    result_paths
    return (result_paths,)


@app.cell(hide_code=True)
def hints_paths():
    L.hints(
        "two-paths",
        [
            "Steps 3 and 4 of the recipe: `t` reaches `z` through `u` and through `v`.",
            "`v` is $e$ to the power $2t$: by the chain rule, its derivative is the outer derivative at the inner value, times the derivative of $2t$. Then each path is two local derivatives multiplied.",
        ],
        why="At $t = 1$: $u = 1$ and $v = e^2 \\approx 7.389$. The path through $u$ gives $7.389 \\cdot 2$, the path through $v$ gives $1 \\cdot 2 \\cdot 7.389$: together about 29.556.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The partial derivatives you measured in [Two parameters: the gradient](/course/open/math-gradient?lang=en) are now exact formulas, by the rules of [Derivative rules](/course/open/math-derivative-rules?lang=en) with the other variables held fixed. And when a value reaches the output along several paths, the multivariable chain rule adds them up. In the next lesson, *Gradients of matrix expressions: follow the shapes*, the variables become whole vectors and matrices at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_partial, result_gradient, result_paths):
    L.wrap_up(choice_partial, result_gradient, result_paths)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
