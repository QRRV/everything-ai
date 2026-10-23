# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-missing-values. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-missing-values", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two tables into one: merge](/course/open/data-merge?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_left():
    quick_left = L.choice(
        "quick-left",
        question="`weather` has the 7 days of a week. The stand was closed on Monday, so `sales` has 6 rows. What does `pd.merge(weather, sales, on=\"day\", how=\"left\")` have in the column `sold` on Monday's row?",
        options={
            "zero": "`0`",
            "right": "`NaN`",
            "none": "Nothing: there is no row for Monday",
        },
        correct="right",
        why={
            "zero": "merge does not know that the stand was closed. What does a left join put where the right table has no match? See the table of new words in [Two tables into one: merge](/course/open/data-merge?lang=en).",
            "none": "That is what an inner join does. Which rows does a left join keep? See the table of new words in [Two tables into one: merge](/course/open/data-merge?lang=en).",
            "right": "A left join keeps every row of `weather`, and puts `NaN` where `sales` has no match: a missing value that the merge itself made.",
        },
    )
    quick_left
    return (quick_left,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="Two columns both have a mean of 10. The standard deviation of the first is 1, of the second 4. What does that say?",
        options={
            "larger": "The second column has the larger values",
            "right": "The values of the second column lie further from their mean",
            "more": "The second column has more values",
        },
        correct="right",
        why={
            "larger": "Both have the same mean. What does the standard deviation measure? See the table of new words in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "more": "The standard deviation does not count values. What does it measure? See the table of new words in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "right": "The standard deviation says how far the values typically lie from the mean: 1 in the first column, 4 in the second.",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The left join of [Two tables into one: merge](/course/open/data-merge?lang=en) left `NaN` on Monday, and the station of [Asking questions of data](/course/open/data-explore?lang=en) missed two temperatures and wrote -999 for the wind of one day. pandas skips such gaps in a mean, but many models stop with an error at a `NaN`. So before a model sees the data, you decide for each gap: leave it out, or fill something in.

    **An everyday example.** Two students missed a test. Do you leave them out of the class average, give them the average, or give them a 0? Each choice gives another average or spread, and which one is fair depends on why they missed it.
    """)
    return


@app.cell(hide_code=True)
def predict_fill():
    guess_fill = L.guess(
        "fill",
        question="The station measured 28 of the month's 30 temperatures: their mean is 9.66 degrees, their standard deviation 2.18. You fill in the 2 missing days with that mean. What happens to the mean and the standard deviation of all 30?",
        options={
            "same": "Both stay the same",
            "smaller": "The mean stays the same; the standard deviation gets smaller",
            "larger": "The mean stays the same; the standard deviation gets larger",
        },
        reveal="**The mean stays at 9.66, and the standard deviation falls to 2.10.** Two days at the mean do not move it, but they add no distance from it, so the typical distance of the 30 values shrinks: the month looks steadier than the measurements show.",
    )
    guess_fill
    return (guess_fill,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A gap can be left out or filled in. `dropna` leaves out every row with a gap, or every column with one. `fillna` fills in a value you know, such as 0 ice creams on a day the stand was closed, or, when you do not know it, a guess from the column, such as its mean or median: that is called imputation. A guess looks just like a measurement afterwards, so you first mark the gaps, in a column of their own.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **na_values** | the codes that `read_csv` should read as missing, such as -999 | `pd.read_csv(path, na_values=[-999])` |
        | **dropna** | leaves out every row that has a gap; with `axis="columns"`, every column that has one | `df.dropna()` |
        | **fillna** | puts a value in every gap, and gives back the filled column or table | `df["temp"].fillna(0)` |
        | **imputation** | filling in a gap with a reasonable guess, such as the mean or the median of its column | `fillna(df["temp"].mean())` |
        | **a new column** | `df["name"] = ...` adds a column to a table, as `d[key] = value` adds a key to a dictionary | `df["temp_missing"] = ...` |
        | **missing indicator** | an extra column that is `True` where a value was missing, so you can still tell a guess from a measurement, and a model can learn whether being missing matters | `df["temp_missing"] = df["temp"].isna()` |

        Below, the month of the station: a dot per day at its temperature. Suppose the thermometer had failed on more days: the slider sets how many of the 30 days are missing, starting at the 2 that really are. The open dots are the missing days, filled in with the mean of the measured days, on the dashed line.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lost_cell():
    def _frame(k):
        k = int(k)
        n = C.numbers(k)
        _mean, _drop, _fill = L.number(n["mean"]), L.number(n["drop"]), L.number(n["fill"])
        return (C.lost_picture(k, {"title": "A month with gaps", "day": "day", "temp": "temperature (°C)",
                                   "alt": f"The temperature of the 30 days of the month, a dot per day. {30 - k} days are measured, as filled dots; the {k} missing days are open dots on a dashed line at the mean of the measured days, {_mean} degrees."}),
                f"{k} of the 30 days missing. Dropped: {n['left']} days left, a standard deviation of {_drop}. Filled in with the mean of the measured days, {_mean} degrees: 30 days, a standard deviation of {_fill}.")

    explore_lost = L.explore("lost", "days missing", _frame, start=2, values=C.LOST, role="",
                             task="Slide from 2 to 20 missing days. What happens to the standard deviation when you drop them, and when you fill them in with the mean?")
    explore_lost
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Dropping keeps about the spread of the days that are left, here between 2.1 and 2.5 degrees at every stop, but with fewer days: had other days failed, the spread could have come out further from 2.18. Filling in with the mean keeps all 30 days and the mean, but every filled-in day sits on the mean and adds no spread: with 20 days missing, the standard deviation is 1.37, against 2.46 for the 10 measured days.
        """),
        L.watch_out("`dropna` and `fillna` give back a new table or column; `df` itself stays as it was. To keep a filled-in column, put it back: `df[\"temp\"] = df[\"temp\"].fillna(...)`."),
        mo.md(r"""
        ### Recipe

        To deal with the gaps in a table:

        1. Make every gap a `NaN`, also a code such as -999: `pd.read_csv(path, na_values=[-999])`.
        2. Count the gaps per column, `df.isna().sum()`, and ask why each is missing.
        3. Do you know the value, such as 0 ice creams on a closed day? Fill it in: `fillna(0)`.
        4. If not: drop the rows with a gap, `df.dropna()`, when they are few; drop the columns with a gap, `df.dropna(axis="columns")`, when they are mostly empty; or mark the gaps, `df["temp_missing"] = df["temp"].isna()`, and fill in the mean or the median of the column.
        5. Compare `describe()` before and after: `count`, `mean`, `std` and `50%`, the median.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The month of the weather station of [Asking questions of data](/course/open/data-explore?lang=en), and the week of the stand of [Two tables into one: merge](/course/open/data-merge?lang=en): both made up by the course. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_missing():
    import pandas as pd

    # 1. the month of the station, with the code -999 read as missing
    _df = pd.read_csv("lessons/data-explore/station.csv", na_values=[-999])
    # 2. the gaps per column
    print(_df.isna().sum())
    # 3. the stand's week: it was closed on Monday, so it sold 0 ice creams
    _weather = pd.read_csv("lessons/data-pandas/weather.csv")
    _sales = pd.read_csv("lessons/data-merge/sales.csv")
    _week = pd.merge(_weather, _sales, on="day", how="left")
    print(_week["sold"].fillna(0))
    # 4. drop the rows with a gap, or the columns with one: the rows and columns that are left
    print(_df.dropna().shape, _df.dropna(axis="columns").shape)
    # 4. or mark the missing temperatures, then fill them in with the mean
    _df["temp_missing"] = _df["temp"].isna()
    _filled = _df["temp"].fillna(_df["temp"].mean())
    print(_df[_df["temp_missing"]])
    # 5. the temperatures before and after
    print(_df["temp"].describe())
    print(_filled.describe())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With -999 read as missing, the wind has a gap too, on day 17. On the stand's Monday, `fillna(0)` puts 0 where the left join had `NaN`: the stand was closed, so 0 is not a guess. `dropna()` keeps 27 of the 30 days, and the three days with a gap take their other measurements with them; `dropna(axis="columns")` keeps all 30 days, but without `temp` and `wind`. Filled in with the mean, the temperature has a `count` of 30 and the same mean, 9.66, but its `std` falls from 2.18 to 2.10 and its median from 10.75 to 10.25. pandas divides by one less than the number of values for `std`, as the *Watch out* of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) said some tools do.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_filled():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The wind of day 17

        The box reads the month of the station as `df`, with -999 read as missing: the wind has a gap on day 17. Make `filled`: the column `wind`, its gap filled in with the median wind. Replace the `...`; the last line is `filled`, so the check sees it. The check also tries the other month.
        """),
        L.task("Make `filled`: the wind of `df`, its gap filled in with the median wind."),
    ])
    return


@app.cell(hide_code=True)
def exercise_filled():
    box_filled = L.code_box("filled", lines=6)
    box_filled
    return (box_filled,)


@app.cell(hide_code=True)
def check_filled(box_filled):
    result_filled = L.check("filled", box_filled.value)
    result_filled
    return (result_filled,)


@app.cell(hide_code=True)
def hints_filled():
    L.hints(
        "filled",
        [
            "Step 4 of the recipe, as the mean in the worked example, but with the median and the column `wind`.",
            "`fillna` puts the value between its parentheses in the gap. pandas has a method for the median, as it has `.mean()` for the mean.",
            "`filled = df[\"wind\"].fillna(df[\"wind\"].___())`.",
        ],
        why="`df[\"wind\"].median()` is the median of the 29 measured wind speeds, 21, and `df[\"wind\"].fillna(...)` puts it in the place of the gap of day 17.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Mark, then fill in

        Finish `fill_flag(path, column)`: it reads the CSV file at `path` as `df`, with -999 read as missing, and gives back `df`. Replace the `...` with two lines: one that makes a missing-indicator column named `f"{column}_missing"`, then one that fills in the gaps of the column `column` with its median. The check calls `fill_flag("lessons/data-explore/station.csv", "temp")`, and also tries other files and columns.
        """),
        L.task("Make `fill_flag(path, column)` add a missing-indicator column, then fill in the column with its median."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flag():
    box_flag = L.code_box("fill-flag", lines=8)
    box_flag
    return (box_flag,)


@app.cell(hide_code=True)
def check_flag(box_flag):
    result_flag = L.check("fill-flag", box_flag.value)
    result_flag
    return (result_flag,)


@app.cell(hide_code=True)
def hints_flag():
    L.hints(
        "fill-flag",
        [
            "Step 4 of the recipe, in its order: first mark the gaps, then fill them in.",
            "Two lines of the form `df[name] = ...`: a new column `f\"{column}_missing\"` that is `True` at every gap, then the filled-in column, back in `df[column]`.",
            "`df[f\"{column}_missing\"] = df[column].isna()`, then `df[column] = df[column].fillna(df[column].___())`.",
        ],
        why="`df[column].isna()` is `True` at every gap, and as the column `f\"{column}_missing\"` it keeps where they were, also after `df[column] = df[column].fillna(df[column].median())` has filled them in.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_discount():
    mo.md(r"""
    ### Exercise 3. Why is it missing?
    """)
    return


@app.cell(hide_code=True)
def choice_discount_cell():
    choice_discount = L.choice(
        "discount",
        question="A webshop has a table of orders, with a column `discount` in euros. It is empty for every order without a discount code. What do you do with those gaps?",
        options={
            "mean": "Fill them in with the mean discount of the other orders",
            "right": "Fill them in with 0",
            "drop": "Drop the orders without a discount",
        },
        correct="right",
        why={
            "mean": "That gives orders without a code a discount they never had. What was the discount of an order without a code?",
            "drop": "Then the table holds only orders with a discount, and their mean discount looks as if every order had one. What was the discount of an order without a code?",
            "right": "No code, so no discount: when you know why a value is missing, it is not unknown. Fill in what it was, 0, as for the stand on the day it was closed.",
        },
    )
    choice_discount
    return (choice_discount,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Asking questions of data](/course/open/data-explore?lang=en) found gaps and promised you would deal with them; the left join of [Two tables into one: merge](/course/open/data-merge?lang=en) made gaps of its own, and one of them got the 0 it stood for. Filling in with the mean shrinks the standard deviation of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en). Later, a model learns from one part of the data and is tested on another; the mean or median you fill in must then come only from the part it learns from, which a later lesson takes up. In the next lesson, *Two columns together: an association is not a cause*, you look at two columns at once, and at what a third can do to them.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_filled, result_flag, choice_discount):
    L.wrap_up(result_filled, result_flag, choice_discount)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
