# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-matrix-multiply. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("math-matrix-multiply", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Many features: the dot product](/course/open/math-dot-product?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="What is the dot product of $(1, 2)$ and $(3, 4)$?",
        options={
            "products": "$(3, 8)$",
            "all": "10",
            "right": "11",
        },
        correct="right",
        why={
            "products": "Those are the products. What is the last step of a dot product? See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "all": "That adds up all four numbers. What happens to the numbers in the same place first? See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "right": "1 × 3 + 2 × 4 = 3 + 8 = 11.",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def quick_check_np_dot():
    quick_np_dot = L.choice(
        "quick-np-dot",
        question="What does `np.dot(w, x)` do that `w * x` does not?",
        options={
            "sum": "It adds up the products",
            "bias": "It adds the bias term",
            "nothing": "Nothing: they give the same",
        },
        correct="sum",
        why={
            "bias": "The bias term is added by hand, after the dot product. See the recipe in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "nothing": "`w * x` gives an array, `np.dot(w, x)` one number. See the worked example in [Many features: the dot product](/course/open/math-dot-product?lang=en).",
            "sum": "`w * x` multiplies the numbers in the same place; `np.dot` also adds those products up, to one number.",
        },
    )
    quick_np_dot
    return (quick_np_dot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model makes a prediction for every example, and a dataset has many. A dot product per example, in a loop, is slow and takes a lot of code. With all examples in one matrix, one multiplication gives every prediction at once. Each layer of a neural network, later in the course, starts with exactly this.

    **An everyday example.** A shop has a table of baskets: one row per customer, one column per product, and in each cell how many they bought. Multiply each row by the price list and add up, and you have every customer's total at once.
    """)
    return


@app.cell(hide_code=True)
def predict_shape():
    guess_shape = L.guess(
        "shape",
        question="`@` is Python's sign for matrix multiplication. `X` holds 8 houses, with 3 features each: shape (8, 3). `w` holds 3 weights: shape (3,). What do you think `X @ w` gives?",
        options={
            "one": "One number",
            "eight": "8 numbers: one for each house",
            "matrix": "A 3 by 3 matrix",
        },
        reveal="**8 numbers, one for each house.** `@` takes the dot product of each row of `X`, one house, with `w`. 8 rows give 8 dot products: shape (8,).",
    )
    guess_shape
    return (guess_shape,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Put the examples in a matrix $\mathbf{X}$, one row per example and one column per feature. Matrix multiplication takes the dot product of every row with the weights: one prediction per example, all at once.
        """),
        L.formula(r"[[pred:\hat{\mathbf{y}}]] = \mathbf{X} [[param:\mathbf{w}]] + [[param:b]]"),
        mo.md(r"""
        This works for a matrix of weights too, with one column per output: every row of the first matrix times every column of the second. The shapes must fit: the number of columns of the first must equal the number of rows of the second. Then $(n, k)$ times $(k, m)$ gives $(n, m)$. With a vector it works the same: $(n, k)$ times $(k,)$ gives $(n,)$, as `w` acts as one column.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **matrix multiplication** | | each number of the result is the dot product of a row of the first matrix with a column of the second | `X @ w` |
        | **shape rule** | | $(n, k)$ times $(k, m)$ gives $(n, m)$: the two inner numbers must be equal, and the outer ones remain | |
        | $\mathbf{X}$ | capital X | all examples as a matrix, one row per example and one column per feature; shape (number of examples, $d$) | `X` |
        | $\hat{\mathbf{y}}$ | y hat | the predictions of all examples, one per row | `y_pred` |
        """),
        L.watch_out("`@` is Python's sign for matrix multiplication, also for a matrix times a vector. It is not `*`: on two arrays of the same shape, `*` multiplies the numbers in the same place and adds nothing up."),
        draw.array_grids([("X", [[str(_v) for _v in _row] for _row in C.HOUSES], ["shape (4, 3)", "the third row: one house"], {(2, 0), (2, 1), (2, 2)}),
                          ("w", [[str(_v) for _v in C.W]], ["shape (3,)"], {(0, 0), (0, 1), (0, 2)}),
                          ("X @ w", [["169", "250", "315", "126"]], ["shape (4,)", "the third number: 315"], {(0, 2)})],
                         title="Four houses at once"),
        mo.md(r"""
        Four houses, with area, rooms and garden, and the weights of [Many features: the dot product](/course/open/math-dot-product?lang=en). The third row of `X` times `w` is 110 × 2 + 5 × 15 + 1 × 20 = 315: the third number of `X @ w`. Each row gives its own dot product.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make the predictions for many examples at once:

    1. Put the examples in a matrix `X`, one row per example, with the features in the same order as the weights.
    2. Check the shapes: the inner numbers must be equal, such as (4, 3) and (3,).
    3. Multiply with `@`: each row of `X` with `w` gives one dot product.
    4. Add the bias term: NumPy adds it to every prediction at once.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The predictions for the four houses of the picture, with $b = 30$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_matmul():
    import numpy as np

    # 1. one row per house: area, rooms, garden
    _X = np.array([[62, 3, 0], [85, 4, 1], [110, 5, 1], [48, 2, 0]])
    _w = np.array([2, 15, 20])
    _b = 30
    # 2. the shapes: the inner numbers are both 3
    print(_X.shape, _w.shape)
    # 3. every dot product at once
    print(_X @ _w)
    # 4. plus the bias term: the predictions
    _y_pred = _X @ _w + _b
    print(_y_pred, _y_pred.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_X` has shape (4, 3) and `_w` shape (3,): the 3's fit. `_X @ _w` gives the four dot products, 169, 250, 315 and 126, and `+ _b` adds 30 to each: the predictions are 199, 280, 345 and 156 thousand euros, shape (4,). With 8000 houses, the code would be exactly the same.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which shape?
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "result-shape",
        question="`A` has shape (5, 3) and `B` has shape (3, 2). What is the shape of `A @ B`?",
        options={
            "inner": "(3, 3)",
            "right": "(5, 2)",
            "error": "No shape: they do not fit",
        },
        correct="right",
        why={
            "inner": "Those are the inner numbers. Which numbers remain? See the shape rule in *The idea*.",
            "error": "Look at the inner numbers: the 3 of `A` and the 3 of `B`. Are they equal? See the shape rule in *The idea*.",
            "right": "The inner numbers are both 3, so they fit, and the outer ones remain: (5, 2).",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_grades():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. All students at once

        The model of [Many features: the dot product](/course/open/math-dot-product?lang=en) predicts a grade from hours of study and hours of sleep. Now `X` holds three students, one row each: shape (3, 2). `w` holds the two weights, shape (2,), and `b` is the bias term. Put the predictions for all three in `y_pred`, with `@`. Replace the `...`. The last line is `y_pred`, so the check sees it. The check also tries other students.
        """),
        L.task("Put the predicted grades of all students in `y_pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grades():
    box_grades = L.code_box("all-grades", lines=8)
    box_grades
    return (box_grades,)


@app.cell(hide_code=True)
def check_grades(box_grades):
    result_grades = L.check("all-grades", box_grades.value)
    result_grades
    return (result_grades,)


@app.cell(hide_code=True)
def hints_grades():
    L.hints(
        "all-grades",
        [
            "Steps 3 and 4 of the recipe: multiply with `@`, then add the bias term.",
            "Which comes first, the matrix with the students or the weights? Check the shapes: (3, 2) and (2,).",
            "`y_pred = X @ ___ + ___`.",
        ],
        why="`X @ w` takes the dot product of each student's row with `w`: 5.4, 7.1 and 3.7. `+ b` adds 1 to each: 6.4, 8.1 and 4.7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The shape rule as a function

        Finish `result_shape(a, b)`. It gets two shapes, each a tuple of two numbers, such as `(4, 2)`. It gives back the shape of the matrix multiplication, as a tuple, or `False` when the shapes do not fit. Replace the `...`. The check calls `result_shape((4, 2), (2, 3))`: that should give `(4, 3)`. It also tries other shapes, some that do not fit.
        """),
        L.task("Make `result_shape` apply the shape rule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rule():
    box_rule = L.code_box("shape-rule", lines=6)
    box_rule
    return (box_rule,)


@app.cell(hide_code=True)
def check_rule(box_rule):
    result_rule = L.check("shape-rule", box_rule.value)
    result_rule
    return (result_rule,)


@app.cell(hide_code=True)
def hints_rule():
    L.hints(
        "shape-rule",
        [
            "Two steps: first check whether the shapes fit, then give back the new shape.",
            "`a[0]` and `a[1]` are the rows and columns of the first shape, `b[0]` and `b[1]` those of the second. Which two are the inner numbers?",
            "`if a[1] != ___:` with `return False`, and otherwise `return (a[0], ___)`.",
        ],
        why="The shapes fit when the number of columns of `a` equals the number of rows of `b`: `a[1] == b[0]`. Then the result has the rows of `a` and the columns of `b`: `(a[0], b[1])`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Many features: the dot product](/course/open/math-dot-product?lang=en) you worked out one dot product for one example; matrix multiplication works out one per row, for every example at once, and the shape rule says when two shapes fit. In the next lesson, *Flipping a matrix: the transpose*, you swap the rows and columns of a matrix, which is often what makes two shapes fit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_grades, result_rule):
    L.wrap_up(choice_shape, result_grades, result_rule)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
