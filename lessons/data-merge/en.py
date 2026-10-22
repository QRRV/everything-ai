# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-merge. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-merge", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One number per group: groupby](/course/open/data-groupby?lang=en) and [Dictionaries](/course/open/py-dicts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_size():
    quick_size = L.choice(
        "quick-size",
        question="`df` is the month of the weather station, a row per day. What does `df.groupby(\"sky\").size()` give?",
        options={
            "kinds": "How many kinds of sky there are",
            "right": "For each kind of sky, how many days it has",
            "temps": "For each kind of sky, how many temperatures it has, without the missing ones",
        },
        correct="right",
        why={
            "kinds": "That would be one number. `groupby` gives a number per group: what does `size` count in each group? See the table of new words in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "temps": "That is what `count` does, on the column `temp`. `size` looks at no column: what does it count? See the table of new words in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "right": "`size` counts the rows of each group, and here every row is a day.",
        },
    )
    quick_size
    return (quick_size,)


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="`sold = {\"sat\": 95}`, and then `sold[\"sat\"] = 80`. What is `sold` now?",
        options={
            "both": "`{\"sat\": 95, \"sat\": 80}`",
            "right": "`{\"sat\": 80}`",
            "old": "`{\"sat\": 95}`",
        },
        correct="right",
        why={
            "both": "Can a key be in a dictionary twice? See the table of new words in [Dictionaries](/course/open/py-dicts?lang=en).",
            "old": "The second line gives the key `\"sat\"` a value. What happens to the old one? See the guess in [Dictionaries](/course/open/py-dicts?lang=en).",
            "right": "A key is in a dictionary only once, so the second line replaces its value: you look up `\"sat\"` and find one answer.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The weather of a week is in one table; the ice creams a beach stand sold each day are in another. To see whether warm days sell more, you need both on one row per day: by hand, you would look up each day of one table in the other. A merge does that for every row at once.

    **An everyday example.** A class list has student numbers and names; the test results have student numbers and grades. To put each name next to its grade, you match them on the student number.
    """)
    return


@app.cell(hide_code=True)
def predict_rows():
    guess_rows = L.guess(
        "rows",
        question="`weather` has 7 days, Monday to Sunday. The stand was closed on Monday, so `sales` has 6 rows, Tuesday to Sunday. How many rows do you think `pd.merge(weather, sales, on=\"day\")` has?",
        options={
            "six": "6",
            "seven": "7",
            "thirteen": "13",
        },
        reveal="**6.** merge puts rows side by side where the day is the same. Monday is only in `weather`, so it is left out, without a word: an inner join, what merge does unless you ask otherwise. 13 would be the tables stacked, not merged.",
    )
    guess_rows
    return (guess_rows,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        `pd.merge` puts the rows of two tables side by side where they have the same value in a column both have: the key, here `day`. Every pair of rows with the same key becomes one row, with the columns of both; this is called a join. What happens to a row without a match depends on `how`: an inner join leaves it out; a left join keeps every row of the left table, with `NaN` in the columns of the other table. A key that is in the right table twice matches twice, so its row comes out twice.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **join** | two tables made into one by matching rows on a column both have; pandas calls it merge | `pd.merge(weather, sales, on="day")` |
        | **key** | the column the rows are matched on; as in a dictionary, its value says which rows belong together | `on="day"` |
        | **left and right table** | the first and the second table you give merge | `pd.merge(left, right, on=...)` |
        | **inner join** | keeps only rows whose key is in both tables; the default | `how="inner"` |
        | **left join** | keeps every row of the left table, with `NaN` where the right table has no match | `how="left"` |

        Below, the week of weather is next to a list of the stand with Saturday's line typed twice. The slider looks up the days of `weather` one by one: a line joins a day to each row of the list with that day, and each line becomes a row of both joins; a day without a line gets a row only in the left join. It starts with all 7 days looked up.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lookup_cell():
    _once = "`{day}` is in the list once: each join gets one row."
    _say = [
        "No day looked up yet: both joins are empty.",
        "`mon` is not in the list: the inner join gets no row, the left join one with `NaN`.",
        *[_once.format(day=_day) for _day in ("tue", "wed", "thu", "fri")],
        "`sat` is in the list twice: each join gets two rows.",
        "`sun` is in the list once. All 7 days looked up: the inner join has 7 rows, the left join 8.",
    ]

    def _frame(days):
        days = int(days)
        return (C.lookup(days, {"title": "Saturday typed twice", "inner": "inner join", "left": "left join", "row": "row", "rows": "rows",
                                "alt": "The week of weather and the stand's list, with Saturday twice; a line joins each day looked up to each row of the list with that day. Then the inner and the left join of those days. Days looked up: {k} of 7. Rows in the inner join: {n}; in the left join: {m}."}),
                _say[days])

    explore_lookup = L.explore("lookup", "days", _frame, start=7, values=list(range(8)), role="",
                               task="Slide to 0, then step through the days. Which day gives the inner join no row, and which gives it two?")
    explore_lookup
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Monday gives the inner join no row and the left join one with `NaN`; Saturday gives each join two. So the inner join has 7 rows, as many as `weather`, and yet Monday is missing and Saturday is in twice. Only the left join, with 8, shows that a row was added.
        """),
        L.watch_out("merge says nothing when a row finds no match, or finds two. Count the rows before and after every merge."),
        mo.md(r"""
        ### Recipe

        To combine two tables:

        1. Find the key: the column both tables have. Its values must be written the same way in both.
        2. Count the rows of both tables: `len(weather)` and `len(sales)`. For a table, `len` gives the number of rows, as it gives the number of items of a list.
        3. Merge, and choose `how`: an inner join, the default, or `how="left"` to keep every row of the left table.
        4. Count the rows again: a left join should have as many as the left table. More means a key is in the right table more than once; `sales.groupby("day").size()` shows which.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The week of weather of [Tables with pandas](/course/open/data-pandas?lang=en) and the ice creams the stand sold that week: both made up by the course. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_merge():
    import pandas as pd

    # 1. the two tables; both have the column day
    _weather = pd.read_csv("lessons/data-pandas/weather.csv")
    _sales = pd.read_csv("lessons/data-merge/sales.csv")
    print(_sales)
    # 2. the number of rows of both, before the merge
    print(len(_weather), len(_sales))
    # 3. an inner join, and a left join
    _both = pd.merge(_weather, _sales, on="day")
    print(_both)
    _every = pd.merge(_weather, _sales, on="day", how="left")
    print(_every)
    # 4. the number of rows after; then the list with Saturday typed twice
    print(len(_both), len(_every))
    _twice = pd.read_csv("lessons/data-merge/sales-twice.csv")
    print(len(pd.merge(_weather, _twice, on="day", how="left")))
    print(_twice.groupby("day").size())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The inner join has 6 rows, without Monday; the left join 7, with `NaN` for Monday's ice creams, so pandas writes `sold` as decimals: 61.0 instead of 61. Weather and sales now share one row per day: the warmest, Saturday at 16.5 degrees, sold the most, 95; the wettest, Wednesday with 8 millimeters of rain, the fewest, 18. With Saturday typed twice, the left join has 8 rows, and `size` shows why: `sat` has 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_week():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Every day of week 2

        The box reads the second week of weather as `weather`, and what the stand sold that week as `sales`. The stand was closed on Monday and on Friday. Make `week`: a left join of `weather` and `sales` on `day`. Replace the `...`. The last line is `week`, so the check sees it. The check also tries week 1.
        """),
        L.task("Make `week`: a left join of `weather` and `sales` on `day`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_week():
    box_week = L.code_box("week-2", lines=7)
    box_week
    return (box_week,)


@app.cell(hide_code=True)
def check_week(box_week):
    result_week = L.check("week-2", box_week.value)
    result_week
    return (result_week,)


@app.cell(hide_code=True)
def hints_week():
    L.hints(
        "week-2",
        [
            "Step 3 of the recipe, as the left join in the worked example.",
            "`weather` goes first, as the left table; the key after `on=`, the kind of join after `how=`.",
            "`week = pd.merge(___, ___, on=___, how=___)`.",
        ],
        why="`pd.merge(weather, sales, on=\"day\", how=\"left\")` keeps all 7 days of `weather`, with the ice creams of `sales` next to them: 33, 58 and 71 from Tuesday to Thursday, 30 and 49 at the weekend, `NaN` on Monday and Friday.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_extra():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Count the rows

        Finish `extra_rows(left_path, right_path, key)`. It reads two CSV files as `left` and `right`. It gives back how many rows a left join of `left` and `right` on `key` adds: the rows after the merge minus the rows before. Replace the `...`. The check calls `extra_rows("lessons/data-pandas/weather.csv", "lessons/data-merge/sales-twice.csv", "day")`: that should give 1. It also tries other tables.
        """),
        L.task("Make `extra_rows(left_path, right_path, key)` give back how many rows a left join adds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_extra():
    box_extra = L.code_box("extra-rows", lines=8)
    box_extra
    return (box_extra,)


@app.cell(hide_code=True)
def check_extra(box_extra):
    result_extra = L.check("extra-rows", box_extra.value)
    result_extra
    return (result_extra,)


@app.cell(hide_code=True)
def hints_extra():
    L.hints(
        "extra-rows",
        [
            "Steps 3 and 4 of the recipe: the left join, then the rows after minus the rows before.",
            "`key` holds the name of the column, such as `\"day\"`: use it after `on=` without quotes.",
            "`merged = pd.merge(left, right, on=___, how=___)`, and `return len(___) - len(___)`.",
        ],
        why="`pd.merge(left, right, on=key, how=\"left\")` keeps every row of `left`, and a row whose key is in `right` twice comes out twice. So `len(merged) - len(left)` counts the rows it added: 1 for Saturday typed twice.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fewer():
    mo.md(r"""
    ### Exercise 3. Fewer rows than you expected
    """)
    return


@app.cell(hide_code=True)
def choice_fewer_cell():
    choice_fewer = L.choice(
        "fewer-rows",
        question="`weather` has 7 rows, Monday to Sunday. Another stand was open every day: its `sales` has 7 rows. Yet `pd.merge(weather, sales, on=\"day\")` has only 6 rows. What is the most likely cause?",
        options={
            "spelled": "One day is written another way in `sales`, such as `Sat` for `sat`, so it finds no match",
            "missing": "One day in `sales` has `NaN` for the ice creams",
            "twice": "One day is in `sales` twice",
        },
        correct="spelled",
        why={
            "missing": "merge only looks at the key: a missing value in another column does not drop a row. Which rows does an inner join leave out? See *The idea*.",
            "twice": "A key that is in the right table twice gives more rows, not fewer: look at `sat` in *The idea*. Which rows does an inner join leave out?",
            "spelled": "An inner join drops a row whose key is not in the other table, and for merge `Sat` and `sat` are two keys. `sales[\"day\"].value_counts()` would show the odd one out, as in [Asking questions of data](/course/open/data-explore?lang=en).",
        },
    )
    choice_fewer
    return (choice_fewer,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Dictionaries](/course/open/py-dicts?lang=en) you looked up a value by its key; a merge looks up each row of the left table in the right one, and finds one row, none or several. In [One number per group: groupby](/course/open/data-groupby?lang=en), `size` counted the rows per group; here it finds a key that is in a table twice. In [Asking questions of data](/course/open/data-explore?lang=en), `Sunny` and `sunny` were one sky written two ways; in a merge, that loses a row without a word. In the next lesson, *Where does this data come from?*, you ask who made a dataset and what you may do with it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_week, result_extra, choice_fewer):
    L.wrap_up(result_week, result_extra, choice_fewer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
