# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-matrix-rank. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-matrix-rank", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) and [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="What shape does `B @ C` have, with `B` of shape (5, 2) and `C` of shape (2, 5)?",
        options={
            "inner": "(2, 2)",
            "error": "None: it stops with an error",
            "right": "(5, 5)",
        },
        correct="right",
        why={
            "inner": "The inner numbers, 2 and 2, must match and then drop out; the outer ones remain. See [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "error": "The inner numbers are both 2, so the shapes fit. See [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "right": "(5, 2) @ (2, 5): the inner 2s match, and the outer 5 and 5 remain.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_column():
    quick_column = L.choice(
        "quick-column",
        question="`c` is an array of shape (3, 1), a column. What is the shape of `c.T`?",
        options={
            "right": "(1, 3): a row",
            "same": "(3, 1): `.T` changes nothing here",
            "flat": "(3,): the brackets drop away",
        },
        correct="right",
        why={
            "same": "`.T` changes nothing only for an array with one number in its shape, such as (3,). This one has two. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "flat": "The transpose swaps the two numbers of the shape; it never drops one. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "right": "The transpose swaps rows and columns: (3, 1) becomes (1, 3).",
        },
    )
    quick_column
    return (quick_column,)


@app.cell(hide_code=True)
def text_why():
    mo.md(f"""
    ## Why this matters

    A large language model has weight matrices of 4,096 by 4,096: {C.grouped(4096 * 4096, L.lang)} numbers each, and dozens of them. To adapt such a model to a new task, you need not change all those numbers freely: a change whose rows are all combinations of a few rows is often enough. Such a change can be stored as two thin matrices, with a small fraction of the numbers. Later in the course you adapt a model this way, with LoRA.

    **An everyday example.** A paint shop sells hundreds of colours, but mixes all of them from a few base pigments. Write down the pigments and one recipe per colour, and you have every colour, with far less to keep.
    """)
    return


@app.cell(hide_code=True)
def predict_rank():
    guess_rank = L.guess(
        "rank-guess",
        question="A matrix has three rows: `[1, 2, 3]`, `[2, 4, 6]` and `[1, 0, 1]`. How many really different directions do you think these rows have?",
        options={
            "three": "3: there are three rows",
            "right": "2: one row adds nothing new",
            "one": "1: they all start with a small number",
        },
        reveal="**2.** The second row is exactly twice the first: it points the same way, only longer, so it adds no new direction. The third row cannot be made from the first by scaling, so it does add one. Two independent rows: the rank of this matrix is 2.",
    )
    guess_rank
    return (guess_rank,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A row is **independent** of other rows when you cannot make it by scaling those rows and adding them up. The **rank** of a matrix is its number of independent rows: how many directions it really has. It is always also the number of independent columns, so it is at most the smaller of the two numbers in its shape, and `A.T` has the same rank as `A`. In code, `np.linalg.matrix_rank(A)` gives it.

        Multiply a tall, thin matrix $\mathbf{B}$ of shape $(d, r)$ by a wide, thin matrix $\mathbf{C}$ of shape $(r, d)$, and you get a square matrix $\mathbf{A} = \mathbf{B}\mathbf{C}$ of shape $(d, d)$. Its rank is at most $r$. You can see why by writing the product as a sum of $r$ pieces: column $i$ of $\mathbf{B}$ times row $i$ of $\mathbf{C}$. In LoRA, $\mathbf{B}$ and $\mathbf{C}$ are what the model learns, so here they are orange, and so are $\mathbf{A}$ and its pieces, which they make.
        """),
        L.formula(r"[[param:\mathbf{A}]] = [[param:\mathbf{B}]][[param:\mathbf{C}]] = [[param:\mathbf{b}_1 \mathbf{c}_1]] + [[param:\mathbf{b}_2 \mathbf{c}_2]] + \dots + [[param:\mathbf{b}_r \mathbf{c}_r]]"),
        mo.md(f"""
        Each piece $\\mathbf{{b}}_i \\mathbf{{c}}_i$ is a column of shape $(d, 1)$ times a row of shape $(1, d)$: a full $(d, d)$ matrix whose rows are all multiples of the same row $\\mathbf{{c}}_i$, so it has rank 1, unless it is all zeros. Each piece adds at most one direction, so $r$ pieces give at most $r$.

        A matrix with a rank far below its size is **low-rank**. Stored as $\\mathbf{{B}}$ and $\\mathbf{{C}}$, it needs $2 \\cdot d \\cdot r$ numbers instead of $d \\cdot d$. For $d$ = 4,096 and $r$ = 8 that is {C.grouped(2 * 4096 * 8, L.lang)} numbers instead of {C.grouped(4096 * 4096, L.lang)}: less than half a percent.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **independent** | | linearly independent: a row you cannot make by scaling other rows and adding them up | |
        | **rank** | | the number of independent rows of a matrix, which is also its number of independent columns | `np.linalg.matrix_rank(A)` |
        | **low-rank** | | with a rank far below its size; such a matrix can be stored as two thin matrices | `B @ C` |
        | $\\mathbf{{b}}_i \\mathbf{{c}}_i$ | b i c i | column $i$ of $\\mathbf{{B}}$ times row $i$ of $\\mathbf{{C}}$: a matrix of rank 1, or 0 if it is all zeros | `B[:, 0:1] @ C[0:1, :]` |
        """),
        L.watch_out("Rank is about directions, not about size or zeros: a matrix full of large numbers can have rank 1, and the matrix with the rows `[1, 0]` and `[0, 1]`, half zeros, has rank 2. And `matrix_rank` counts directions up to rounding: a row that is a combination of others up to a tiny float error still counts as dependent."),
        mo.md(r"""
        The picture: a tall matrix $\mathbf{B}$ of shape (8, 2) times a wide matrix $\mathbf{C}$ of shape (2, 8) gives a square matrix $\mathbf{A}$ of shape (8, 8), each drawn to scale.
        """),
        mo.Html(C.thin_picture({"title": "Two thin matrices make one square one", "numbers": "{n} numbers",
                                "note": "A has 64 numbers, but rank at most {r}:\nB and C hold only 32.",
                                "alt": "Three boxes drawn to scale: B, tall and thin, (8, 2), 16 numbers; times C, wide and thin, (2, 8), 16 numbers; gives A, square, (8, 8), 64 numbers. Below: A has 64 numbers, but rank at most 2: B and C hold 32."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make and check a low-rank matrix:

    1. Choose the rank $r$, and make a tall matrix $\mathbf{B}$ of shape $(d, r)$ and a wide matrix $\mathbf{C}$ of shape $(r, d)$.
    2. Multiply them: $\mathbf{A} = \mathbf{B}\mathbf{C}$, of shape $(d, d)$.
    3. The same matrix is the sum of $r$ pieces: column $i$ of $\mathbf{B}$ times row $i$ of $\mathbf{C}$.
    4. Check the rank with `np.linalg.matrix_rank`, and count the numbers: $2 \cdot d \cdot r$ for $\mathbf{B}$ and $\mathbf{C}$, against $d \cdot d$ for $\mathbf{A}$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A matrix of shape (6, 6) and rank 2. In `B[:, 0:1]`, the `0:1` keeps the column as a column, of shape (6, 1); `C[0:1, :]` keeps the row as a row, of shape (1, 6). The comments with `#` say which step of the recipe each part is. Steps 3 and 4 are not finished: run the cell as it is, then complete them where the comment says, and run it again.
    """)
    return


@app.cell
def example_rank():
    import numpy as np

    # 1. r = 2: B is (6, 2), C is (2, 6)
    _B = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [0, 3], [1, 2]])
    _C = np.array([[1, 2, 0, 1, 0, 3], [0, 1, 1, 0, 2, 1]])
    # 2. the product: a matrix of shape (6, 6)
    _A = _B @ _C
    print(_A)
    # 3. the same matrix as pieces of rank 1: the first piece is here; add the second, with 1:2
    _pieces = _B[:, 0:1] @ _C[0:1, :]
    print(_pieces - _A)
    # 4. the rank, and the numbers: A has 6 * 6; add how many B and C hold together
    print(np.linalg.matrix_rank(_A), 6 * 6)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\mathbf{A}$ has 6 rows, but its rank is 2: the third row, `[1, 3, 1, 1, 2, 4]`, is the first plus the second, and the others are combinations of the first two as well. With the first piece alone, the difference is not zero yet: it is minus the second piece, the part still missing. With the second piece added, it is all zeros: the two pieces add up to exactly $\mathbf{A}$. $\mathbf{B}$ and $\mathbf{C}$ hold 24 numbers against 36; the bigger the matrix, the larger the saving.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The rank of a matrix
    """)
    return


@app.cell(hide_code=True)
def choice_rank_cell():
    choice_rank = L.choice(
        "rank-3",
        question="What is the rank of the matrix with the rows `[1, 0, 1]`, `[0, 1, 1]` and `[1, 1, 2]`?",
        options={
            "three": "3",
            "one": "1",
            "right": "2",
        },
        correct="right",
        why={
            "three": "The third row is the first plus the second, so it adds no new direction.",
            "one": "The first two rows point in different directions: neither is a multiple of the other.",
            "right": "The first two rows are independent, and the third is their sum: two directions.",
        },
    )
    choice_rank
    return (choice_rank,)


@app.cell(hide_code=True)
def text_exercise_numbers():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How many numbers?

        A weight matrix need not be square: it has the shape $(d_\text{out}, d_\text{in})$. `thin_numbers(d_out, d_in, r)` should give how many numbers $\mathbf{B}$, of shape $(d_\text{out}, r)$, and $\mathbf{C}$, of shape $(r, d_\text{in})$, hold together. Replace the `...`. The check calls `thin_numbers(5, 4, 2)`, which should give `18`, and tries other shapes.
        """),
        L.task("Make `thin_numbers` count the numbers of `B` and `C` together."),
    ])
    return


@app.cell(hide_code=True)
def exercise_numbers():
    box_numbers = L.code_box("thin-numbers", lines=4)
    box_numbers
    return (box_numbers,)


@app.cell(hide_code=True)
def check_numbers(box_numbers):
    result_numbers = L.check("thin-numbers", box_numbers.value)
    result_numbers
    return (result_numbers,)


@app.cell(hide_code=True)
def hints_numbers():
    L.hints(
        "thin-numbers",
        [
            "Step 4 of the recipe, for a matrix that is not square.",
            "A matrix of shape $(n, m)$ holds $n \\cdot m$ numbers. Count `B` and `C` each by its own shape, and add.",
        ],
        why="`B` is (5, 2), 10 numbers; `C` is (2, 4), 8 numbers: together 18, against 20 for the full (5, 4) matrix.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pieces():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A matrix from pieces

        `from_pieces(cols, rows)` gets $r$ columns of shape $(d, 1)$ and $r$ rows of shape $(1, d)$, and should give the sum of the $r$ pieces, column $i$ times row $i$, as in step 3 of the recipe. Write the body of the function. The check calls it with two columns of 3 numbers and two rows of 3 numbers, and tries other pieces too.
        """),
        L.task("Make `from_pieces` give the sum of all pieces, each a column times a row."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pieces():
    box_pieces = L.code_box("from-pieces", lines=9)
    box_pieces
    return (box_pieces,)


@app.cell(hide_code=True)
def check_pieces(box_pieces):
    result_pieces = L.check("from-pieces", box_pieces.value)
    result_pieces
    return (result_pieces,)


@app.cell(hide_code=True)
def hints_pieces():
    L.hints(
        "from-pieces",
        [
            "Step 3 of the recipe: piece `i` is `cols[i]` times `rows[i]`. Start with the first piece, and add the others in a loop over `range(1, len(cols))`.",
            "A column times a row is (d, 1) @ (1, d): a (d, d) matrix. Which order gives that shape?",
        ],
        why="The first piece is `[1, 0, 2]` as a column times `[1, 2, 0]` as a row; the second is `[0, 1, 1]` times `[0, 1, 3]`. Added, they give a (3, 3) matrix of rank 2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A matrix product of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) can be read as a sum of pieces, each a column times a row; [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en) swaps rows and columns, and leaves the rank the same. A matrix built from $r$ such pieces has rank at most $r$, and two thin matrices store it with far fewer numbers. In the next lesson, *Common distributions*, you leave matrices for a while and describe chance with formulas.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rank, result_numbers, result_pieces):
    L.wrap_up(choice_rank, result_numbers, result_pieces)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
