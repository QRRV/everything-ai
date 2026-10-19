# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson np-indexing. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-indexing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="What is the shape of `np.array([[1, 2, 3, 4], [5, 6, 7, 8]])`?",
        options={
            "flat": "`(8,)`",
            "tall": "`(4, 2)`",
            "wide": "`(2, 4)`",
        },
        correct="wide",
        why={
            "flat": "A list of lists gives a matrix, with two numbers in its shape. See the picture in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "tall": "In a shape, the rows come first. How many inner lists are there? See the table of new words in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "wide": "2 inner lists, so 2 rows, each of 4 numbers: 4 columns.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_dtype():
    quick_dtype = L.choice(
        "quick-dtype",
        question="What is the dtype of `np.array([1, 2.5])`?",
        options={
            "float": "`float64`",
            "int": "`int64`",
            "both": "`int64` for the 1, and `float64` for the 2.5",
        },
        correct="float",
        why={
            "float": "All numbers of an array share one type. One float is enough: the 1 becomes `1.0`.",
            "int": "The 2.5 is not a whole number. What happens to the type of the other numbers? See the guess in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "both": "An array has one type for all its numbers. See the table of new words in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
        },
    )
    quick_dtype
    return (quick_dtype,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You rarely use all of your data at once. You want one example out of a table, so one row; or one feature of every example, so one column; or only the examples that meet a condition, such as every photo of a 7. Later in the course, you do this all the time: to split your data into a part to learn from and a part to test on, or to see which predictions were wrong. With NumPy, each of these takes one short line, without a loop.

    **An everyday example.** A spreadsheet with a filter. You click a column to see all its values, or you filter the rows, and only the ones you asked for stay in view.
    """)
    return


@app.cell(hide_code=True)
def predict_pick():
    guess_pick = L.guess(
        "pick",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. What do you think `m[1, 2]` gives?",
        options={
            "two": "`2`",
            "six": "`6`",
            "five": "`5`",
        },
        reveal="**`6`.** The first number picks the row, the second the column, and both start at 0. Row 1 is `[4, 5, 6]`, and in it, column 2 is `6`.",
    )
    guess_pick
    return (guess_pick,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Square brackets after an array pick a part of it, as with a list. In a matrix, you give a row and a column, with a comma between them. Instead of a number, you can also give a slice, or a mask: an array of `True` and `False` that keeps the numbers where it is `True`.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **`m[row, column]`** | one number: first the row, then the column, both counted from 0 | `m[1, 2]` gives `6` |
    | **`m[row]`** | one whole row, as a vector | `m[0]` gives `array([1, 2, 3])` |
    | **`:`** | the whole direction: all rows, or all columns | `m[:, 1]` gives the column `array([2, 5])` |
    | **slice** | a part of a direction, as with a list | `m[0, 1:3]` gives `array([2, 3])` |
    | **comparison on an array** | compares every number by itself, and gives an array of `True` and `False` of the same shape | `m > 2` gives `array([[False, False, True], [True, True, True]])` |
    | **boolean mask** | an array of `True` and `False` between square brackets: it keeps the numbers where it is `True`, row by row, as a vector | `m[m > 2]` gives `array([3, 4, 5, 6])` |

    *Boolean* is the full name of a bool: `True` or `False`. A mask picks numbers from all over the array, so the result has no rows and columns any more: it is a vector. And one number from an array shows on the last line of a cell as `np.int64(6)`: NumPy's own kind of whole number. `print` shows just `6`.
    """)
    return


@app.cell(hide_code=True)
def picture_select():
    mo.vstack([
        draw.array_grids(C.picture(["one number", "a column", "with a mask"], "gives"), indices=True,
                         title="Three ways to pick from m"),
        mo.md(r"""
        The small numbers above and to the left are the indexes: columns above, rows to the left. The boxes with a thick border are what each selection picks; the rest goes gray. `m[:, 1]` takes column 1 from every row; `m[m > 2]` takes every number above 2.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("The row comes first, then the column, and both start at 0. For a column, you need the `:` for all rows: `m[:, 1]`. `m[1]` is row 1."),
        mo.md(r"""
        ### Recipe

        To pick a part of an array:

        1. One number: `m[row, column]`.
        2. A row: `m[row]`; a column: `m[:, column]`.
        3. A part of a row or a column: a slice, as with a list, such as `m[0, 1:3]`.
        4. The numbers that meet a condition: make a mask with a comparison, such as `m > 2`, and put it between the square brackets: `m[m > 2]`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The rain from [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en): a row per week, a column per day, from Monday to Thursday. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_rain():
    import numpy as np

    _rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])
    # 1. one number: week 2 is row 1, Tuesday is column 1
    print(_rain[1, 1])
    # 2. a row and a column: all of week 1, and every Monday
    print(_rain[0])
    print(_rain[:, 0])
    # 3. a part: week 1, from Tuesday to Wednesday
    print(_rain[0, 1:3])
    # 4. the amounts above 2: first the mask, then the amounts it picks
    _wet = _rain > 2
    print(_wet)
    print(_rain[_wet])
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The cell prints `0`: on the Tuesday of week 2, it did not rain. Then week 1, `[0 3 5 1]`, and the Mondays, `[0 2 6]`: a column comes out as a vector, on one line. `_rain[0, 1:3]` is `[3 5]`: Tuesday and Wednesday of week 1. The mask has the shape of `_rain`, with `True` where it rained more than 2 mm. And `_rain[_wet]` gives those amounts, row by row: `[3 5 4 6]`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_column():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which line takes this part?
    """)
    return


@app.cell(hide_code=True)
def choice_part_cell():
    choice_part = L.choice(
        "which-part",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. Which line gives `array([5, 6])`?",
        options={
            "rows": "`m[1:3]`",
            "part": "`m[1, 1:3]`",
            "short": "`m[1, 1:2]`",
        },
        correct="part",
        why={
            "rows": "With one slice between the brackets, you slice the rows: rows 1 and 2, and `m` only has row 1 of those. So you get all of row 1, as a matrix. Look at step 3 of the recipe again.",
            "part": "Row 1 is `[4, 5, 6]`, and the slice `1:3` takes its columns 1 and 2: `5` and `6`.",
            "short": "A slice stops just before its second number, so `1:2` takes only column 1: `array([5])`. Where should it stop?",
        },
    )
    choice_part
    return (choice_part,)


@app.cell(hide_code=True)
def text_exercise_wednesdays():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Every Wednesday

        The box has the rain again, `rain`, with shape `(3, 4)`: a row per week, a column per day, from Monday to Thursday. Take out the rain of every Wednesday, as `wednesdays`. Replace the `...`. The last line is `wednesdays`, so the check sees it.
        """),
        L.task("Make `wednesdays` the column of Wednesdays, taken out of `rain`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wednesdays():
    box_wednesdays = L.code_box("wednesdays", lines=6)
    box_wednesdays
    return (box_wednesdays,)


@app.cell(hide_code=True)
def check_wednesdays(box_wednesdays):
    result_wednesdays = L.check("wednesdays", box_wednesdays.value)
    result_wednesdays
    return (result_wednesdays,)


@app.cell(hide_code=True)
def hints_wednesdays():
    L.hints(
        "wednesdays",
        [
            "Step 2 of the recipe: a column.",
            "A column needs all rows, so `:` comes first; after the comma comes the number of the column. Monday is column 0.",
            "`wednesdays = rain[:, ___]`.",
        ],
        why="`:` takes all rows, the three weeks, and `2` takes column 2, Wednesday, from each week: `array([5, 0, 0])`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_wet():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The wet days

        Finish `wet_days(rain, limit)`: it gives back every amount in `rain` that is more than `limit`, as an array, row by row. Here `rain` is a list of lists, and the function already turns it into an array, `grid`. Replace the `...`. The check calls `wet_days(rain, 2)` with the rain of exercise 2, as a list of lists: that should give `array([3, 5, 4, 6])`.
        """),
        L.task("Make `wet_days(rain, limit)` give back the amounts above `limit`, with a boolean mask."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wet():
    box_wet = L.code_box("wet-days", lines=8)
    box_wet
    return (box_wet,)


@app.cell(hide_code=True)
def check_wet(box_wet):
    result_wet = L.check("wet-days", box_wet.value)
    result_wet
    return (result_wet,)


@app.cell(hide_code=True)
def hints_wet():
    L.hints(
        "wet-days",
        [
            "Step 4 of the recipe, as at the end of the worked example.",
            "First the mask: a comparison of `grid` with `limit`. Then put the mask between the square brackets of `grid`, and give that back.",
            "`wet = grid ___ limit`, and then `return grid[___]`.",
        ],
        why="`grid > limit` compares every amount with `limit`, and gives `True` where it is more. `grid[wet]` keeps those amounts, row by row, as a vector.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Lists](/course/open/py-lists?lang=en) you took an item out of a list with its index, and a part with a slice; an array works the same way, now with a row and a column. In [Making choices: if and else](/course/open/py-if?lang=en) a comparison gave one `True` or `False`; on an array, it gives one for every number, and that is a mask. In the next lesson, *Calculating without loops*, you calculate with whole arrays at once, the way a comparison already works on every number.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_part, result_wednesdays, result_wet):
    L.wrap_up(choice_part, result_wednesdays, result_wet)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
