# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-explore. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-explore", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Tables with pandas](/course/open/data-pandas?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`df` has a column `wind`. Which line keeps the rows with more wind than 20?",
        options={
            "mask": "`df[df[\"wind\"] > 20]`",
            "column": "`df[\"wind\"] > 20`",
            "name": "`df[\"wind\" > 20]`",
        },
        correct="mask",
        why={
            "column": "That gives `True` or `False` for every row. Where does that mask go, to keep the rows? See the recipe in [Tables with pandas](/course/open/data-pandas?lang=en).",
            "name": "That compares the text `\"wind\"` with 20. What should the mask compare? See the table of new words in [Tables with pandas](/course/open/data-pandas?lang=en).",
            "mask": "`df[\"wind\"] > 20` is `True` for the rows with more wind than 20; between the square brackets of `df`, it keeps those rows.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_middle():
    quick_middle = L.choice(
        "quick-middle",
        question="In a column of numbers, the mean is 30 and the median 25. What does that point to?",
        options={
            "low": "A few very small values",
            "same": "Nothing: those two always differ this much",
            "high": "A few very large values",
        },
        correct="high",
        why={
            "low": "Very small values would pull the mean down, below the median. Which way is the mean pulled here? See the picture in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "same": "When the values lie evenly around the middle, the mean and the median are about the same. What pulls them apart? See the table of new words in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "high": "The mean is pulled up, above the median. A few very large values, such as an outlier, do that.",
        },
    )
    quick_middle
    return (quick_middle,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model learns from whatever data you give it, mistakes included. A missing value, a sensor that writes -999 when it fails, or `Sunny` next to `sunny`: a model cannot tell that those are wrong, and learns them. So before you build anything, you look at the data: what is in it, what is missing, and what is strange.

    **An everyday example.** Buying a second-hand bike: before you pay, you look at it from all sides, try the brakes, and ask about anything that looks odd.
    """)
    return


@app.cell(hide_code=True)
def predict_skip():
    guess_skip = L.guess(
        "skip",
        question="A weather station measured the temperature every day for 30 days, but on two days nothing was recorded. What do you think `df[\"temp\"].mean()` does with those two days?",
        options={
            "skip": "It leaves them out, and gives the mean of the other 28",
            "zero": "It counts them as 0 degrees",
            "error": "It stops with an error",
        },
        reveal="**It leaves them out.** pandas skips missing values in `.mean()`, `.sum()` and the like, without a word: the mean is taken over 28 days. That is handy, but you only know it happened if you look.",
    )
    guess_skip
    return (guess_skip,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Exploratory data analysis is a set of questions you ask every new dataset, before you build anything with it. How big is it? What is in each column? What is missing? What cannot be right? pandas answers most of them in one line each. A missing value is a place where nothing was recorded: pandas shows it as `NaN`, and skips it in sums and means. Strange values come in a few kinds: impossible numbers, such as a negative amount of rain; codes that mean "no measurement", such as -999; and one thing written in two ways, such as `sunny` and `Sunny`.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **exploratory data analysis** | looking at a new dataset with summaries and charts, before any model, to learn what is in it and to spot problems | |
    | **missing value** | a place where nothing was recorded; pandas shows it as `NaN` | `NaN` |
    | **describe** | per column of numbers: how many values there are (`count`), the mean, the standard deviation, the smallest value, the 0.25, 0.5 and 0.75 quantiles (shown as `25%`, `50%` and `75%`), and the largest value | `df.describe()` |
    | **isna** | `True` for every missing value and `False` for the rest: a table as big as `df` | `df.isna()` |
    | **`.sum()` and `.mean()` of a mask** | `.sum()` counts the `True` values per column, as `True` counts as 1 and `False` as 0; `.mean()` gives the fraction of `True` values, from 0 to 1 | `df.isna().sum()` |
    | **value_counts** | how often each value of a column appears, the most frequent first | `df["sky"].value_counts()` |
    """)
    return


@app.cell(hide_code=True)
def picture_month():
    mo.vstack([
        mo.Html(C.picture({"title": "A month of the weather station", "temp": "temperature (°C)", "wind": "wind (km/h)", "day": "day",
                           "alt": "Two line charts of the 30 days. The temperature runs between 5.5 and 12 degrees, with gaps on day 9 and day 22. The wind runs between 7 and 33 kilometers per hour, but on day 17 it drops to -999, so that all other days look flat."})),
        mo.md(r"""
        The month of the station, day by day. In the temperature, two days are missing: the line has gaps there. The wind stays between 7 and 33 kilometers per hour, but on day 17 it dives to -999, so far down that all other days look flat. One look at a chart shows both.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("pandas skips missing values in `.mean()`, `.sum()` and `describe`, without a word. A mean can be over fewer values than you think: look at `count`."),
        mo.md(r"""
        ### Recipe

        To explore a new table:

        1. Look at its size and its first rows: `df.shape` and `df.head()`.
        2. Summarize the columns of numbers: `df.describe()`. Look at `count` (is something missing?), at `min` and `max` (can that be?), and at the mean against the median, `50%`.
        3. Count the missing values per column: `df.isna().sum()`.
        4. Count the values of each column of text, such as `df["sky"].value_counts()`. Is one thing written in two ways?
        5. Write down what is strange, as questions for whoever made the data, before you build anything with it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up weather station measured every day for a month: the temperature in degrees Celsius, the rain in millimeters, the wind in kilometers per hour, the direction of the wind, and how the sky looked. The file is `lessons/data-explore/station.csv`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_station():
    import pandas as pd

    # 1. its size and its first rows
    _df = pd.read_csv("lessons/data-explore/station.csv")
    print(_df.shape)
    print(_df.head())
    # 2. a summary of each column of numbers
    print("describe:")
    print(_df.describe())
    # 3. the missing values per column
    print("missing:")
    print(_df.isna().sum())
    # 4. how often each kind of sky appears
    print("sky:")
    print(_df["sky"].value_counts())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The table has 30 rows and 6 columns. In `describe`, `count` is 30 for every column of numbers except `temp`, which has 28: two temperatures are missing, and `isna().sum()` says the same. The smallest wind is -999, and that makes the mean wind -12.1 kilometers per hour, which cannot be: wind is never below 0. The median wind, `50%`, is 21: one value pulls the mean far away from it. `describe` only shows the columns of numbers; `value_counts` shows that the sky was `sunny` on 12 days and `Sunny` on 1: the same sky, written another way. Step 5, the questions to write down: why are days 9 and 22 missing? Is -999 the station's code for "no measurement"? Should `Sunny` count as `sunny`?
    """)
    return


@app.cell(hide_code=True)
def text_exercise_code():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A strange maximum
    """)
    return


@app.cell(hide_code=True)
def choice_code_cell():
    choice_code = L.choice(
        "error-code",
        question="A fitness watch writes the heart rate, in beats per minute, into a column `heart_rate`. In `df.describe()`, the largest value of `heart_rate` is `999`. What is most likely?",
        options={
            "workout": "A very hard workout",
            "code": "A code the watch writes when it measures nothing",
            "typo": "A typing error for 99",
        },
        correct="code",
        why={
            "workout": "Can a heart beat 999 times a minute, even in the hardest workout? What else could a number like 999 be?",
            "typo": "A heart rate of 99 is possible, but would a typing error give exactly 999, all nines? What else could such a number be?",
            "code": "No heart beats 999 times a minute, and a number of all nines is what many devices write when they measure nothing, like the -999 of the station. Treat it as a missing value, and ask whoever made the data.",
        },
    )
    choice_code
    return (choice_code,)


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What is missing?

        The box reads the month of the station as `df`. Make `missing`: the fraction of missing values in each column, from 0 to 1. Replace the `...`. The last line is `missing`, so the check sees it. The check also tries another month.
        """),
        L.task("Make `missing`: the fraction of missing values in each column of `df`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing", lines=6)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "Two steps: first mark every missing value, then turn each column of marks into one number.",
            "`df.isna()` marks the missing values with `True`. What in the table of new words gives the fraction of `True` values in each column?",
            "`missing = df.___().___()`.",
        ],
        why="`df.isna()` is `True` where a value is missing, and `.mean()` of that mask gives the fraction of `True` values in each column: 2 of 30, about 0.067, for `temp`, and 0 for the other columns.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How often does each value appear?

        Finish `counts(path, column)`. It reads the CSV file at `path` as `df`, and gives back how often each value of the column named `column` appears. Replace the `...`. The check calls `counts("lessons/data-explore/station.csv", "sky")`, and also tries another month and another column.
        """),
        L.task("Make `counts(path, column)` give back how often each value of the column `column` appears."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("counts", lines=7)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "counts",
        [
            "Step 4 of the recipe: first take the column, then count its values.",
            "`column` holds the name of a column, such as `\"sky\"`: use it between the square brackets of `df`, without quotes.",
            "`return df[___].___()`.",
        ],
        why="`df[column]` takes the column whose name is in `column`, and `.value_counts()` counts how often each of its values appears.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Tables with pandas](/course/open/data-pandas?lang=en) you read a table and selected rows from it; now you ask it questions. In [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) you summarized one column; `describe` does that for every column of numbers at once. Later in the course, the data a model trains on is checked like this first, and missing values and codes such as -999 are dealt with before a model sees them. In the next lesson, *Where does this data come from?*, you ask the questions that no table answers: who made the data, how, and what you may do with it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_code, result_counts, result_missing):
    L.wrap_up(choice_code, result_missing, result_counts)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
