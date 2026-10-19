# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson np-axis. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-axis", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en) and [Repeating: for loops](/course/open/py-for?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="A matrix has a row per student and a column per test: 30 students, 4 tests. What is its shape?",
        options={
            "tests": "`(4, 30)`",
            "count": "`(120,)`",
            "students": "`(30, 4)`",
        },
        correct="students",
        why={
            "tests": "In a shape, the rows come first. What is a row here? See the table of new words in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "count": "That is the shape of a vector of 120 numbers; a matrix has two numbers in its shape. See the picture in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "students": "30 rows, one per student, and 4 columns, one per test. The rows come first.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_mean():
    quick_mean = L.choice(
        "quick-mean",
        question="What does this program show?\n\n```python\nnumbers = [4, 8, 6]\ntotal = 0\nfor x in numbers:\n    total = total + x\nprint(total / len(numbers))\n```",
        options={
            "mean": "`6.0`",
            "total": "`18`",
            "last": "`2.0`",
        },
        correct="mean",
        why={
            "mean": "The loop adds 4, 8 and 6: 18. Divided by the 3 numbers, that is 6.0; `/` always gives a float.",
            "total": "That is `total` itself. What happens to it on the last line? See the recipe in [Repeating: for loops](/course/open/py-for?lang=en).",
            "last": "`total` is not just the last number: each round adds to it. See the worked example in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_mean
    return (quick_mean,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A table full of numbers says little until you summarize it: the mean grade of each test, the total rain of each week. Later in the course, you do this all the time. How wrong a model is comes down to the mean of its mistakes over all examples. And a model that recognizes digits gives a row of ten scores per image: its prediction for each image is the digit with the highest score in that row.

    **An everyday example.** A grade sheet. At the bottom of each column, the mean of that test; at the end of each row, the mean of that student.
    """)
    return


@app.cell(hide_code=True)
def predict_axis():
    guess_axis = L.guess(
        "axis",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. What do you think `m.sum(axis=0)` gives?",
        options={
            "columns": "`array([5, 7, 9])`",
            "rows": "`array([ 6, 15])`",
            "all": "`21`",
        },
        reveal="**`array([5, 7, 9])`.** `axis=0` goes down the rows: for each column, it adds the numbers under each other, 1 + 4, 2 + 5 and 3 + 6. The rows disappear, and one number per column is left. So `axis=0` does not mean \"per row\".",
    )
    guess_axis
    return (guess_axis,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    An array can summarize itself: its sum, its mean, or where its largest number is. Without an axis, you get one number for the whole array. With an axis, you choose the direction: the summary goes along that direction, and one number is left for each row or each column.

    ### New words and symbols

    Below, `m` is the matrix from the question above, `[[1, 2, 3], [4, 5, 6]]`. `axis=0` between the parentheses is a named value: the name `axis`, then `=` and the number.

    | Word | What it means | In code |
    |---|---|---|
    | **`.sum()`** | adds up all numbers of the array | `print(m.sum())` shows `21` |
    | **mean** | the sum divided by how many numbers there are; `.mean()` | `print(m.mean())` shows `3.5` |
    | **axis** | one direction of an array: axis 0 runs down the rows, axis 1 runs along each row | |
    | **`axis=0`** | works down the rows: one result per column | `m.sum(axis=0)` gives `array([5, 7, 9])` |
    | **`axis=1`** | works along each row: one result per row | `m.sum(axis=1)` gives `array([ 6, 15])` |
    | **`.argmax()`** | the index of the largest number; with an axis, one index per column or per row | `print(np.array([4, 9, 2]).argmax())` shows `1` |

    `.max()` gives the largest number itself; `.argmax()` gives its place. When the last line of a cell is one number from an array, the cell shows it as `np.int64(21)` or `np.float64(3.5)`; `print` shows just the number.
    """)
    return


@app.cell(hide_code=True)
def picture_axes():
    _rows, _down, _across = C.picture()
    mo.vstack([
        draw.axis_results(_rows, _down, _across, down_label="m.sum(axis=0)", across_label="m.sum(axis=1)",
                          title="The sums of m along each axis"),
        mo.md(r"""
        The dashed boxes are what comes out. `m.sum(axis=0)` goes down the rows, so it gives one sum under each column: `5`, `7` and `9`. `m.sum(axis=1)` goes along each row, so it gives one sum next to each row: `6` and `15`. `m.sum()` adds all of them: `21`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`axis=0` does not mean \"per row\". It is the direction the summary goes in: down the rows, so the rows disappear and one number per column is left. Check it with the shape of the result."),
        mo.md(r"""
        ### Recipe

        To summarize an array:

        1. Choose the summary: `.sum()`, `.mean()` or `.argmax()`.
        2. One number for the whole array: no axis.
        3. One number per column: `axis=0`, down the rows. One number per row: `axis=1`, along each row.
        4. Check the shape of the result: a matrix of shape `(3, 4)` gives 4 numbers with `axis=0`, and 3 with `axis=1`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The grades of three students on four tests: a row per student, a column per test. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_grades():
    import numpy as np

    _grades = np.array([[7, 8, 6, 8], [5, 7, 8, 6], [9, 9, 7, 10]])
    # 1. and 2. the mean of all grades: one number
    print(_grades.mean())
    # 3. the mean of each test: down the rows, axis=0
    print(_grades.mean(axis=0))
    # 3. the mean of each student: along each row, axis=1
    print(_grades.mean(axis=1))
    # 1. and 3. the best student of each test: the index of the highest grade, down the rows
    _best = _grades.argmax(axis=0)
    print(_best)
    # 4. the shape of that result: one number per test
    print(_best.shape)
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The mean of all twelve grades is `7.5`. Per test, down the rows, it is `[7. 8. 7. 8.]`: four numbers, one per column. Per student, along each row, it is `[7.25 6.5  8.75]`: three numbers, one per row. The best student of each test is `[2 2 1 2]`: student 2, the third row, was best in three tests, and student 1 in test 2, the third test. Its shape is `(4,)`: one number per test, as step 4 expects.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weeks():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The rain of each week
    """)
    return


@app.cell(hide_code=True)
def choice_weeks_cell():
    choice_weeks = L.choice(
        "week-totals",
        question="`rain` has a row per week and a column per day: shape `(3, 4)`. Which line gives the total rain of each week?",
        options={
            "down": "`rain.sum(axis=0)`",
            "along": "`rain.sum(axis=1)`",
            "all": "`rain.sum()`",
        },
        correct="along",
        why={
            "down": "That goes down the rows and gives one total per column: per day, 4 numbers. Look at the *Watch out* box above the recipe again.",
            "along": "A week is a row, and `axis=1` adds along each row: 3 totals, one per week.",
            "all": "Without an axis, you get one number: all the rain of the three weeks together. How do you get one per row?",
        },
    )
    choice_weeks
    return (choice_weeks,)


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The mean of each day

        The temperature on four days, each in the morning, at noon and in the evening: `temps` has a row per day, shape `(4, 3)`. Make `day_means`: the mean temperature of each day. Replace the `...`. The last line is `day_means`, so the check sees it.
        """),
        L.task("Make `day_means`: one mean per day, so per row of `temps`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("day-means", lines=6)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("day-means", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "day-means",
        [
            "Step 3 of the recipe: one number per row.",
            "The mean is `.mean()`. A day is a row, and one number per row means going along each row.",
            "`day_means = temps.mean(axis=___)`.",
        ],
        why="`axis=1` goes along each row, so each day keeps one number: the mean of its three temperatures. The first day, 10, 16 and 13, has a mean of 13.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predictions():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. What does the model see?

        A model looks at photos and gives each photo three scores: for a cat, a dog and a bird, in that order. Its prediction is the animal with the highest score.

        Finish `predictions(scores)`: `scores` is a list with a row of scores per photo, and the function already turns it into an array. The function gives back, for each photo, the index of its highest score: `0` for cat, `1` for dog, `2` for bird. It must work for any number of photos and animals; when two scores are equally high, it gives the first. Replace the `...`. The check calls `predictions([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])`: that should give `array([1, 0, 2, 2])`.
        """),
        L.task("Make `predictions(scores)` give back the index of the highest score of each photo."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predictions():
    box_predictions = L.code_box("predictions", lines=7)
    box_predictions
    return (box_predictions,)


@app.cell(hide_code=True)
def check_predictions(box_predictions):
    result_predictions = L.check("predictions", box_predictions.value)
    result_predictions
    return (result_predictions,)


@app.cell(hide_code=True)
def hints_predictions():
    L.hints(
        "predictions",
        [
            "Each row is one photo. You want, for each row, the place of its highest score.",
            "The place of the largest number is `.argmax()`. One result per row means going along each row.",
            "`return scores.argmax(axis=___)`.",
        ],
        why="`scores.argmax(axis=1)` goes along each row and gives the index of its highest score: `1` for the first photo, a dog, because 0.7 is its highest score.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Repeating: for loops](/course/open/py-for?lang=en) you added numbers one by one in a loop; `.sum()` and `.mean()` do it in one step. In [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en) the shape told you the rows and the columns; an axis says which of the two disappears. In the next lesson, *Broadcasting*, you calculate with arrays of different shapes, such as every grade minus the mean of its test.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weeks, result_means, result_predictions):
    L.wrap_up(choice_weeks, result_means, result_predictions)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
