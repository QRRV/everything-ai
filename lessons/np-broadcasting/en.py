# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson np-broadcasting. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-broadcasting", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Calculating without loops](/course/open/np-vectorization?lang=en) and [Sums and means along an axis](/course/open/np-axis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_times():
    quick_times = L.choice(
        "quick-times",
        question="`a = np.array([1, 2])` and `b = np.array([3, 4])`. What does `a * b` give?",
        options={
            "total": "`11`",
            "pairs": "`array([3, 8])`",
            "joined": "`array([1, 2, 3, 4])`",
        },
        correct="pairs",
        why={
            "total": "That would be one number for everything. What does `*` do with two arrays of the same shape? See the table of new words in [Calculating without loops](/course/open/np-vectorization?lang=en).",
            "pairs": "The numbers in the same place go together: 1 times 3, and 2 times 4.",
            "joined": "Arrays are not stuck together by `*`. See the picture in [Calculating without loops](/course/open/np-vectorization?lang=en).",
        },
    )
    quick_times
    return (quick_times,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="`m` has shape `(3, 4)`. What is the shape of `m.mean(axis=0)`?",
        options={
            "rows": "`(3,)`",
            "one": "One number, without a shape",
            "cols": "`(4,)`",
        },
        correct="cols",
        why={
            "rows": "`axis=0` goes down the rows, so the rows disappear. What is left? See the *Watch out* box in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "one": "That is what you get without an axis. See the recipe in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "cols": "Down the rows, the 3 rows disappear, and one mean per column is left: 4 numbers.",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Often you want to subtract one number per column from a whole table, such as every grade minus the mean of its test, so that you can see who did better than the others on each test. Later in the course, a model adds the same numbers to each of the many examples it handles at once. With the rule of this lesson, NumPy does that in one line of code, without copying the small array.

    **An everyday example.** A teacher who adds one point to every grade of a hard test, and two to every grade of a harder one: the same correction for every student, but a different one per test.
    """)
    return


@app.cell(hide_code=True)
def predict_stretch():
    guess_stretch = L.guess(
        "stretch",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])` has shape `(2, 3)`. What do you think `m + np.array([10, 20, 30])` gives?",
        options={
            "rows": "`array([[11, 22, 33], [14, 25, 36]])`",
            "error": "An error: the two shapes differ",
            "first": "Only the first row changes: `array([[11, 22, 33], [4, 5, 6]])`",
        },
        reveal="**`array([[11, 22, 33], [14, 25, 36]])`.** The shapes differ, but they fit: NumPy stretches the vector over the two rows, as if it were there twice, and adds number by number. That is broadcasting.",
    )
    guess_stretch
    return (guess_stretch,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    With two arrays of different shapes, NumPy stretches every length 1, or missing length, to the length of the other array in that place, until both have the same shape, and then calculates number by number. That is broadcasting. It only works when the shapes fit, and the rule for that is short.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **broadcasting** | NumPy stretches each length 1, or missing length, to the other length, without copying anything, and then calculates number by number | `m + np.array([10, 20, 30])` |
    | **the rule** | write the shapes under each other, lined up on the right; each pair must be the same, or one of the two must be 1; a missing length counts as 1 | `(2, 3)` and `(3,)` fit |
    | **a vector** | shape `(3,)` against `(2, 3)`: it goes with every row | `m + np.array([10, 20, 30])` |
    | **a column** | shape `(2, 1)` against `(2, 3)`: one number per row, stretched over the columns | `m + np.array([[100], [200]])` |
    | **`keepdims=True`** | keeps the axis that a summary removes, with length 1, so that one number per row stays a column | `m.sum(axis=1, keepdims=True)` has shape `(2, 1)` |

    When the shapes do not fit, NumPy stops with an error such as `operands could not be broadcast together with shapes (3,4) (3,)`. The operands are the two arrays of the calculation, and their shapes follow, written without spaces. But with a square matrix, such as `(3, 3)`, a vector of shape `(3,)` always fits, even when you meant one number per row: then you get a wrong result, and no error. A column made with `keepdims=True` keeps it right.
    """)
    return


@app.cell(hide_code=True)
def picture_stretch():
    mo.vstack([
        draw.array_grids(C.picture(["m", "v, stretched", "m + v"], ["stretched to (2, 3), not copied", "every row plus v"]),
                         title="Broadcasting: a vector stretched over the rows"),
        mo.md(r"""
        `v` is the vector `[10 20 30]`: the row with the thick border. To fit `m`, NumPy acts as if `v` were there twice: the gray row is the stretched copy, which NumPy never really makes. Then it adds number by number, so every row of `m` gets `10`, `20` and `30` added.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A vector of shape `(2,)` does not fit a matrix of shape `(2, 3)`: lined up on the right, 2 and 3 differ. For one number per row, you need a column, shape `(2, 1)`: use `keepdims=True`."),
        mo.md(r"""
        ### Recipe

        To calculate with two arrays of different shapes:

        1. Write both shapes under each other, lined up on the right.
        2. Compare them pair by pair, from the right. They fit when every pair is the same or has a 1; a missing length counts as 1.
        3. Calculate: NumPy stretches each 1 to the other shape's length in that place.
        4. One number per row against a matrix: make it a column, with `keepdims=True` when it comes from a summary.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The grades of [Sums and means along an axis](/course/open/np-axis?lang=en): a row per student, a column per test. First every grade minus the mean of its test, then every grade minus the mean of its student. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_centered():
    import numpy as np

    _grades = np.array([[7, 8, 6, 8], [5, 7, 8, 6], [9, 9, 7, 10]])
    _test_means = _grades.mean(axis=0)
    # 1. and 2. the shapes: (3, 4) and (4,); from the right, 4 and 4 are the same, and the missing one counts as 1
    print(_grades.shape, _test_means.shape)
    # 3. every grade minus the mean of its test: the means are stretched over the rows
    print(_grades - _test_means)
    # 4. one mean per student, kept as a column, shape (3, 1)
    _student_means = _grades.mean(axis=1, keepdims=True)
    # 1. and 2. (3, 4) and (3, 1): from the right, 4 and 1, and one of them is 1; then 3 and 3, the same
    print(_grades.shape, _student_means.shape)
    # 3. every grade minus the mean of its student: the column is stretched over the columns
    _centered = _grades - _student_means
    print(_centered)
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The shapes `(3, 4)` and `(4,)` fit, so every grade gets the mean of its test taken away: `0` means as good as the mean of that test, and `2` means 2 points better. Student 2, the last row, did better than the mean in three tests. The means per student, with `keepdims=True`, have shape `(3, 1)`: a column. Taken away from the grades, it shows how each grade compares with that student's own mean. Without `keepdims=True`, the means per student would have shape `(3,)`, and `(3, 4)` minus `(3,)` stops with an error: lined up on the right, 4 and 3 differ.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which one fits?
    """)
    return


@app.cell(hide_code=True)
def choice_fits_cell():
    choice_fits = L.choice(
        "which-fits",
        question="`m` has shape `(3, 4)`. Which array can NumPy add to `m`?",
        options={
            "four": "An array of shape `(4,)`",
            "three": "An array of shape `(3,)`",
            "turned": "An array of shape `(4, 3)`",
        },
        correct="four",
        why={
            "three": "Lined up on the right, 4 and 3 differ, and neither is 1. Look at the *Watch out* box above the recipe again.",
            "turned": "From the right: 4 and 3 differ, and neither is 1. Same numbers, but in the wrong places.",
            "four": "From the right, 4 and 4 are the same, and the missing length counts as 1: NumPy stretches it over the 3 rows.",
        },
    )
    choice_fits
    return (choice_fits,)


@app.cell(hide_code=True)
def text_exercise_centered():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Warmer or colder than usual?

        The temperature on four days, in the morning, at noon and in the evening: `temps` has a row per day, shape `(4, 3)`. Make `centered`: every temperature minus the mean of its time of day, so that you see which days were warmer than usual. Replace the `...`. The last line is `centered`, so the check sees it.
        """),
        L.task("Make `centered`: every temperature of `temps` minus the mean of its column."),
    ])
    return


@app.cell(hide_code=True)
def exercise_centered():
    box_centered = L.code_box("centered", lines=6)
    box_centered
    return (box_centered,)


@app.cell(hide_code=True)
def check_centered(box_centered):
    result_centered = L.check("centered", box_centered.value)
    result_centered
    return (result_centered,)


@app.cell(hide_code=True)
def hints_centered():
    L.hints(
        "centered",
        [
            "First the mean of each time of day: one per column. Then take those means away from `temps`.",
            "One mean per column: the summary goes down the rows. Its shape, `(3,)`, fits `(4, 3)` from the right, so NumPy stretches it over the rows.",
            "`centered = temps - temps.mean(axis=___)`.",
        ],
        why="`temps.mean(axis=0)` gives the mean of each time of day, shape `(3,)`. NumPy stretches it over the four rows, so every temperature gets the mean of its own column taken away.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Each town's share

        An election: `votes` has a row per town and a column per party, and holds how many votes each party got there. Finish `shares(votes)`: it gives back, for each town, the share of each party, as an array of the same shape. A share is the votes of the party divided by all votes of that town, so each row adds up to 1. The function already turns `votes` into an array. Replace the `...`. The check calls `shares([[30, 50, 20], [10, 10, 30]])`: that should give `array([[0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])`.
        """),
        L.task("Make `shares(votes)` divide every number by the total of its own row."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=8)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Step 4 of the recipe: one number per row, the total of each town, against a matrix.",
            "The total of each row is a sum along each row. Keep it as a column, so that it fits `votes`; then divide.",
            "`totals = votes.sum(axis=1, ___)`, and then `return votes / ___`.",
        ],
        why="`votes.sum(axis=1, keepdims=True)` gives the total of each town as a column, shape `(rows, 1)`. NumPy stretches it over the columns, so every number is divided by the total of its own town.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Calculating without loops](/course/open/np-vectorization?lang=en) two arrays had to have the same shape; now NumPy stretches the smaller one, when the shapes fit. In [Sums and means along an axis](/course/open/np-axis?lang=en) a summary along an axis made that axis disappear; with `keepdims=True` it stays, with length 1, so the result fits the array it came from. Later in the course, you divide each row of a table of counts by its total, as in exercise 3, to turn counts into chances. In the next lesson, *Random numbers and seeds*, you make arrays full of random numbers, the way a model starts before it has learned anything.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fits, result_centered, result_shares):
    L.wrap_up(choice_fits, result_centered, result_shares)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
