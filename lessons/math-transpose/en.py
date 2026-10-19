# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-transpose. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("math-transpose", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`X` has shape (4, 3) and `w` has shape (3,). What is the shape of `X @ w`?",
        options={
            "inner": "(3,)",
            "same": "(4, 3)",
            "right": "(4,)",
        },
        correct="right",
        why={
            "inner": "That is the inner number. Which number remains? See the shape rule in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "same": "Each row of `X` gives one dot product: one number. How many rows are there? See the worked example in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "right": "The inner numbers are both 3, and the 4 remains: one prediction for each of the 4 rows.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="`A` and `B` both have shape (2, 3). Does `A @ B` work?",
        options={
            "equal": "Yes: the shapes are equal",
            "right": "No: the inner numbers, 3 and 2, differ",
            "result": "Yes: it gives shape (2, 3)",
        },
        correct="right",
        why={
            "equal": "The shape rule does not ask for equal shapes. Which two numbers must be equal? See the shape rule in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "result": "Look at the inner numbers first: the 3 of `A` and the 2 of `B`. See the shape rule in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "right": "The number of columns of `A`, 3, must equal the number of rows of `B`, 2. They do not, so Python stops with an error.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Often there is more than one set of weights: two models, or one model that predicts two things. Each gets its own row of weights in a matrix `W`. To use them, every example, a row of `X`, must meet every row of `W`. But matrix multiplication takes rows of the first times *columns* of the second, so the shapes do not fit. Swap the rows and columns of `W`, and they do. A layer of a neural network in PyTorch, later in the course, stores its weights in exactly this way.

    **An everyday example.** A mark sheet has one row per student and one column per subject. The report form wants one row per subject and one column per student: to fill it in, you turn the sheet around. The marks stay the same; only the rows and columns swap.
    """)
    return


@app.cell(hide_code=True)
def predict_fit():
    guess_fit = L.guess(
        "fit",
        question="Two models, each with 3 weights, are the rows of `W`: shape (2, 3). `X` holds 4 houses: shape (4, 3). Which multiplication do you think works?",
        options={
            "xw": "`X @ W`",
            "wx": "`W @ X`",
            "neither": "Neither of them",
        },
        reveal="**Neither of them.** (4, 3) and (2, 3): the inner numbers are 3 and 2. (2, 3) and (4, 3): 3 and 4. Swap the rows and columns of `W`, and its shape becomes (3, 2): then (4, 3) and (3, 2) fit, and give (4, 2), one prediction per house per model.",
    )
    guess_fit
    return (guess_fit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Transposing a matrix swaps its rows and columns: the first row becomes the first column, the second row the second column, and so on. The result is its transpose. The numbers stay the same, but the shape turns around.
        """),
        L.formula(r"\mathbf{A} \text{ has shape } (n, m) \quad\Longrightarrow\quad \mathbf{A}^\top \text{ has shape } (m, n)"),
        mo.md(r"""
        So with the weights of two models as the rows of $\mathbf{W}$, the predictions of both for every example are $\mathbf{X}\mathbf{W}^\top$: each row of $\mathbf{X}$ times each row of $\mathbf{W}$, now a column.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **transpose** | | the matrix with its rows and columns swapped: the first row becomes the first column, so shape $(n, m)$ becomes $(m, n)$ | `A.T` |
        | $\mathbf{A}^\top$ | A transpose | the transpose of the matrix $\mathbf{A}$ | `A.T` |
        """),
        L.watch_out("`.T` changes nothing on an array with only one number in its shape, such as `w` with shape (3,): it has no rows and columns to swap. `w.T` is just `w`."),
        draw.array_grids([("A", [[str(_v) for _v in _row] for _row in C.A], ["shape (2, 3)", "the first row: 1, 2, 3"], {(0, 0), (0, 1), (0, 2)}),
                          ("A.T", [[str(_row[_j]) for _row in C.A] for _j in range(3)], ["shape (3, 2)", "the first column: 1, 2, 3"], {(0, 0), (1, 0), (2, 0)})],
                         title="Rows become columns"),
        mo.md(r"""
        The first row of `A`, 1, 2 and 3, is the first column of `A.T`, from top to bottom. The second row, 4, 5 and 6, is the second column. The 2 rows of 3 have become 3 rows of 2.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make two shapes fit with a transpose:

    1. Write down both shapes.
    2. Are the inner numbers different? See whether swapping the rows and columns of one of them makes them equal.
    3. Transpose that one with `.T`.
    4. Multiply with `@`, and check the shape of the result: the rows and columns you wanted?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The four houses of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en), with two models: the weights of that lesson, and a second model with other weights, both without a bias term. Each model is a row of `_W`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_transpose():
    import numpy as np

    _X = np.array([[62, 3, 0], [85, 4, 1], [110, 5, 1], [48, 2, 0]])
    _W = np.array([[2, 15, 20], [0.5, 3, 1]])
    # 1. the shapes: the inner numbers are 3 and 2
    print(_X.shape, _W.shape)
    # 2. and 3. the transpose of _W has shape (3, 2): now the 3's fit
    print(_W.T.shape)
    # 4. multiply, and check the shape of the result
    _y_pred = _X @ _W.T
    print(_y_pred)
    print(_y_pred.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_X` has shape (4, 3) and `_W` shape (2, 3): they do not fit. `_W.T` has shape (3, 2), and `_X @ _W.T` gives shape (4, 2): one row per house, one column per model. The first column, 169, 250, 315 and 126, is `X @ w` of the earlier lesson, before the bias term was added; the second column is the second model.

    Without NumPy, a transpose is a loop. The cell below collects the first column of `A` from the picture: the first number of every row.
    """)
    return


@app.cell
def example_column():
    _rows = [[1, 2, 3], [4, 5, 6]]
    _column = []
    for _row in _rows:
        _column.append(_row[0])
    print(_column)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A matrix times its transpose
    """)
    return


@app.cell(hide_code=True)
def choice_gram_cell():
    choice_gram = L.choice(
        "gram",
        question="`A` has shape (5, 2). What is the shape of `A.T @ A`?",
        options={
            "right": "(2, 2)",
            "outer": "(5, 5)",
            "error": "No shape: they do not fit",
        },
        correct="right",
        why={
            "outer": "That is the shape of `A @ A.T`. What is the shape of `A.T`, and which numbers remain? See the recipe.",
            "error": "First write down the shape of `A.T`. Then look at the inner numbers. See the formula in *The idea*.",
            "right": "`A.T` has shape (2, 5). (2, 5) and (5, 2): the inner 5's fit, and the outer numbers remain: (2, 2).",
        },
    )
    choice_gram
    return (choice_gram,)


@app.cell(hide_code=True)
def text_exercise_models():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Two models, all students

        Two models predict a grade from hours of study and hours of sleep. Each model is a row of `W`: the first is the model of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) without its bias term, the second weighs sleep more. `X` holds three students, one row each: shape (3, 2). `W` has shape (2, 2). Put the predictions of both models for all students in `y_pred`, with one row per student and one column per model. Replace the `...`. The last line is `y_pred`, so the check sees it. The check also tries other students.
        """),
        L.task("Put the predictions of both models for all students in `y_pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_models():
    box_models = L.code_box("two-models", lines=8)
    box_models
    return (box_models,)


@app.cell(hide_code=True)
def check_models(box_models):
    result_models = L.check("two-models", box_models.value)
    result_models
    return (result_models,)


@app.cell(hide_code=True)
def hints_models():
    L.hints(
        "two-models",
        [
            "Step 1 of the recipe: write down the shapes of `X` and `W`. Each student must meet each model.",
            "The task asks for one row per student, so `X` comes first. What must the models be, rows or columns?",
            "`y_pred = X @ ___`.",
        ],
        why="`W.T` has shape (2, 2) with one model per column, and `X @ W.T` gives shape (3, 2). The first student, 6 hours of study and 8 of sleep, gets 6 × 0.5 + 8 × 0.3 = 5.4 from the first model and 6 × 0.2 + 8 × 0.6 = 6.0 from the second. `X @ W` also fits, as `W` is square, but then each student meets a column of `W`, not a model.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lists():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A transpose without NumPy

        Finish `transpose(rows)`. It gets a matrix as a list of rows, each a list of numbers, and gives back its transpose, also as a list of rows. Replace the `...`. The check calls `transpose([[1, 2], [3, 4], [5, 6]])`: that should give `[[1, 3, 5], [2, 4, 6]]`. It also tries other matrices.
        """),
        L.task("Make `transpose` swap the rows and columns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lists():
    box_lists = L.code_box("transpose-lists", lines=8)
    box_lists
    return (box_lists,)


@app.cell(hide_code=True)
def check_lists(box_lists):
    result_lists = L.check("transpose-lists", box_lists.value)
    result_lists
    return (result_lists,)


@app.cell(hide_code=True)
def hints_lists():
    L.hints(
        "transpose-lists",
        [
            "The first new row is the first old column: the first number of every row. The second new row holds the second number of every row.",
            "Two loops: an outer one over the places in a row, and an inner one over the rows. How many places does a row have?",
            "Start with `columns = []`. Then `for j in range(___):`, and inside it: a new list `column = []`, a loop `for row in rows:` that appends `row[___]`, and `columns.append(column)`. At the end, `return columns`.",
        ],
        why="For each place `j` in a row, a loop over the rows with `column.append(row[j])` collects the numbers at place `j`: one column. Those columns, in order, are the rows of the transpose. `len(rows[0])` is the number of columns: 2 for the example, which becomes 2 rows.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) two shapes had to fit. The transpose swaps rows and columns, and that is often what makes them fit: with the weights of several models as rows of $\mathbf{W}$, $\mathbf{X}\mathbf{W}^\top$ gives every prediction of every model at once. In the next lesson, *Length, distance and cosine similarity*, you measure how long a vector is, and how far apart or how alike two vectors are.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gram, result_lists, result_models):
    L.wrap_up(choice_gram, result_models, result_lists)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
