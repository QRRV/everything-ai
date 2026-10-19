# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson np-arrays. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-arrays", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Lists](/course/open/py-lists?lang=en), [Using code from modules](/course/open/py-imports?lang=en) and [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question="`days = [3, 0, 5, 2]`. What does `days[1:3]` give?",
        options={
            "three": "`[3, 0, 5]`",
            "two": "`[0, 5]`",
            "rest": "`[0, 5, 2]`",
        },
        correct="two",
        why={
            "three": "Index 0 is the first item, and a slice starts at its first number. See the picture in [Lists](/course/open/py-lists?lang=en).",
            "two": "The slice takes index 1 and 2: `0` and `5`. At index 3 it stops, and that item is not part of it.",
            "rest": "A slice stops at its second number, without that item. See the picture in [Lists](/course/open/py-lists?lang=en).",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def quick_check_import():
    quick_import = L.choice(
        "quick-import",
        question="Which line lets you write `math.sqrt(16)` in your program?",
        options={
            "import": "`import math`",
            "from": "`from math import sqrt`",
            "name": "`math = sqrt`",
        },
        correct="import",
        why={
            "import": "`import math` brings in the module; then the module's name and a dot take `sqrt` out of it.",
            "from": "That brings in only `sqrt`, so you write `sqrt(16)`, without `math.`. See the table of new words in [Using code from modules](/course/open/py-imports?lang=en).",
            "name": "That gives the name `math` a value; it imports nothing. See the recipe in [Using code from modules](/course/open/py-imports?lang=en).",
        },
    )
    quick_import
    return (quick_import,)


@app.cell(hide_code=True)
def quick_check_tuple():
    quick_tuple = L.choice(
        "quick-tuple",
        question="`point = (3, 4)`, and then `x, y = point`. What is `y`?",
        options={
            "first": "`3`",
            "both": "`(3, 4)`",
            "second": "`4`",
        },
        correct="second",
        why={
            "first": "That is `x`: the names get the values in order. See the idea in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "both": "There are two names on the left, so each gets one value of the tuple. See the idea in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "second": "The names get the values in order: `x` the first, `3`, and `y` the second, `4`.",
        },
    )
    quick_tuple
    return (quick_tuple,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    AI calculates with grids of numbers: a black-and-white photo is a grid with one number per pixel, a table of examples has a row per example, and a neural network keeps what it learns in such grids. A Python list is slow with millions of numbers, and it allows anything, such as text between the numbers or rows of different lengths. NumPy keeps numbers in an array: a grid of numbers of one type, with a fixed number of rows and columns.

    **An everyday example.** A spreadsheet with your grades: a row per subject, a column per test, and in every cell a number.
    """)
    return


@app.cell(hide_code=True)
def predict_dtype():
    guess_dtype = L.guess(
        "dtype",
        question="What do you think `np.array([2, 4, 6.5])` holds?",
        options={
            "ints": "Three whole numbers: `2`, `4` and `6`",
            "mixed": "Two whole numbers and one float, as in a list",
            "floats": "Three floats: `2.0`, `4.0` and `6.5`",
        },
        reveal="**Three floats.** All numbers in an array have the same type. As soon as one number is a float, NumPy makes them all floats: `array([2. , 4. , 6.5])`. That type is the array's dtype, here `float64`.",
    )
    guess_dtype
    return (guess_dtype,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    NumPy is a module for calculating with many numbers at once. Its array is a grid of numbers, all of one type. You make an array from a list: a list of numbers gives a vector, and a list of lists gives a matrix, with one inner list per row.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **NumPy** | a module for calculating with many numbers at once; it comes with the course | `import numpy` |
    | **`as`** | imports a module under a short name of your own; NumPy is always imported as `np` | `import numpy as np` |
    | **array** | NumPy's grid of numbers, all of the same type | `np.array([7, 9, 4])` |
    | **list of lists** | a list whose items are lists; for a grid, each inner list is one row | `[[1, 2, 3], [4, 5, 6]]` |
    | **vector** | an array in one direction: numbers one after another | `np.array([7, 9, 4])` |
    | **matrix** | an array in two directions: rows and columns | `np.array([[1, 2, 3], [4, 5, 6]])` |
    | **shape** | how many numbers the array has in each direction, as a tuple; for a matrix, first the rows, then the columns | `m.shape` gives `(2, 3)` |
    | **dtype** | the one type that all numbers of the array share: `int64` for whole numbers, `float64` for floats | `print(m.dtype)` shows `int64` |

    `.shape` and `.dtype` have no parentheses: they are not something the array does, but something it has. The 64 in `int64` and `float64` says how much memory each number takes; you do not need it yet. `print` shows an array without commas, `[7 9 4]`; the last line of a cell shows it with them, `array([7, 9, 4])`.
    """)
    return


@app.cell(hide_code=True)
def picture_arrays():
    mo.vstack([
        draw.array_grids([("a vector", [[str(_x) for _x in C.VECTOR]], [f"shape ({len(C.VECTOR)},)", "3 numbers, one direction"]),
                          ("a matrix", [[str(_x) for _x in _row] for _row in C.MATRIX],
                           [f"shape ({len(C.MATRIX)}, {len(C.MATRIX[0])})", "2 rows, 3 columns"])],
                         title="A vector and a matrix"),
        mo.md(r"""
        A vector goes in one direction, so its shape has one number: `(3,)`. The comma makes it a tuple with one value, and not just a number between parentheses. A matrix goes in two directions, so its shape has two numbers: first the rows, then the columns.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Every row needs the same number of values. With rows of different lengths, NumPy stops with an error: an array is always a full grid."),
        mo.md(r"""
        ### Recipe

        To keep numbers in an array:

        1. Import NumPy once, at the top: `import numpy as np`.
        2. Write the numbers as a list; for a grid, as a list of lists, with one inner list per row.
        3. Give that list to `np.array(...)`, and give the array a name.
        4. Check its `.shape`, first the rows, then the columns, and its `.dtype`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The temperature on two days, in the morning, at noon and in the evening. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_temps():
    # 1. import NumPy, once, at the top
    import numpy as np

    # 2. the numbers as a list of lists: a row per day, a column per time of day
    _rows = [[12.5, 17.0, 14.0], [11.0, 15.5, 13.0]]
    # 3. give the list to np.array
    _temps = np.array(_rows)
    # 4. check the shape and the dtype
    print(_temps.shape)
    print(_temps.dtype)
    # 2. to 4. once more for a vector: one list, of whole numbers
    _noon = np.array([17, 15])
    print(_noon.shape)
    print(_noon.dtype)
    _temps
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the code, the cell shows the matrix itself, with its rows under each other; NumPy writes `17.` for `17.0`. Under that is what the `print` lines show: `(2, 3)`, for 2 rows, one per day, and 3 columns, one per time of day; and `float64`, because all numbers are floats. Then the vector: `(2,)`, two numbers in one direction, and `int64`, because both are whole numbers.

    This cell makes the name `np`, so other cells of this lesson may use it, but not make it again: *one name, one cell*. To try something out, use the scratchpad, with its own `import numpy as np`. The exercise boxes run on their own, so each box has its own `import` line.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Read the shape
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape-of",
        question="What is the shape of `np.array([[1, 2], [3, 4], [5, 6]])`?",
        options={
            "wide": "`(2, 3)`",
            "tall": "`(3, 2)`",
            "flat": "`(6,)`",
        },
        correct="tall",
        why={
            "wide": "That would be 2 rows of 3 numbers. Count the inner lists: each one is a row.",
            "tall": "3 inner lists, so 3 rows, each of 2 numbers: 2 columns. The rows come first.",
            "flat": "That is the shape of a vector of 6 numbers. A list of lists gives a matrix, with two numbers in its shape.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_rain():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A table as an array

        The rain in millimeters, over three weeks, from Monday to Thursday:

        | | Mon | Tue | Wed | Thu |
        |---|---|---|---|---|
        | week 1 | 0 | 3 | 5 | 1 |
        | week 2 | 2 | 0 | 0 | 4 |
        | week 3 | 6 | 1 | 0 | 0 |

        Make the array `rain` from this table, with a row per week. Replace the `...`. The last line is `rain`, so the check sees your array.
        """),
        L.task("Make the array `rain`: a row per week, a column per day."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rain():
    box_rain = L.code_box("rain", lines=5)
    box_rain
    return (box_rain,)


@app.cell(hide_code=True)
def check_rain(box_rain):
    result_rain = L.check("rain", box_rain.value)
    result_rain
    return (result_rain,)


@app.cell(hide_code=True)
def hints_rain():
    L.hints(
        "rain",
        [
            "Steps 2 and 3 of the recipe: first the table as a list of lists, then `np.array`.",
            "Each week is one inner list of 4 numbers, from Monday to Thursday; the three weeks go inside one big list.",
            "`rain = np.array([[0, 3, 5, 1], ___, ___])`.",
        ],
        why="The big list holds three rows, one per week. `np.array` makes a matrix of it, with shape `(3, 4)`: 3 rows, 4 columns. All numbers are whole numbers, so its dtype is `int64`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_describe():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Describe an array

        Finish `describe(table)`. `table` is a list of rows; the function already turns it into an array, `grid`. It should give back one text with the rows, the columns and the dtype of `grid`, such as `"2 by 3, int64"`. Replace the `...`. The check calls `describe([[1, 2, 3], [4, 5, 6]])`: that should give `"2 by 3, int64"`.
        """),
        L.task("Make `describe(table)` give back the rows, the columns and the dtype of the array, as one text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_describe():
    box_describe = L.code_box("describe", lines=8)
    box_describe
    return (box_describe,)


@app.cell(hide_code=True)
def check_describe(box_describe):
    result_describe = L.check("describe", box_describe.value)
    result_describe
    return (result_describe,)


@app.cell(hide_code=True)
def hints_describe():
    L.hints(
        "describe",
        [
            "`grid.shape` is a tuple of two numbers: the rows, then the columns. And `grid.dtype` is the type of the numbers.",
            "Give the two numbers of the shape each a name, as in `x, y = point`; then build the text with an f-string.",
            "`rows, cols = grid.shape`, and then `return f\"{rows} by {cols}, ___\"`.",
        ],
        why="`rows, cols = grid.shape` gives each number of the shape its own name. In an f-string, `{grid.dtype}` becomes `int64` or `float64`. With one float in the table, the whole array is `float64`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Lists](/course/open/py-lists?lang=en) you kept numbers in a list; now a list becomes an array, and a list of lists a matrix. In [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en) a tuple kept values in order; the shape of an array is such a tuple. In [Using code from modules](/course/open/py-imports?lang=en) you imported `math` and `random`; you import NumPy the same way, under the short name `np`. In the next lesson, *Selecting parts of an array*, you take rows, columns and single numbers out of an array.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_describe, result_rain):
    L.wrap_up(choice_shape, result_rain, result_describe)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
