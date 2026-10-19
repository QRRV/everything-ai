# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-pandas. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("data-pandas", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Selecting parts of an array](/course/open/np-indexing?lang=en) and [Reading a text file](/course/open/py-files?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="`a = np.array([3, 8, 1, 6])`. What does `a[a > 4]` give?",
        options={
            "mask": "`array([False, True, False, True])`",
            "places": "`array([1, 3])`",
            "values": "`array([8, 6])`",
        },
        correct="values",
        why={
            "mask": "That is the mask itself, `a > 4`. What happens when it goes between the square brackets of `a`? See the recipe in [Selecting parts of an array](/course/open/np-indexing?lang=en).",
            "places": "Those are the places of the numbers above 4. Between the square brackets of `a`, does a mask give places or numbers? See the table of new words in [Selecting parts of an array](/course/open/np-indexing?lang=en).",
            "values": "The mask is `True` for 8 and 6, so `a[a > 4]` keeps those two numbers.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def quick_check_path():
    quick_path = L.choice(
        "quick-path",
        question="Inside the course folder is a folder `lessons`; inside that, `data-pandas`; and inside that, the file `weather.csv`. Which path leads to that file?",
        options={
            "path": "`\"lessons/data-pandas/weather.csv\"`",
            "name": "`\"weather.csv\"`",
            "back": "`\"weather.csv/data-pandas/lessons\"`",
        },
        correct="path",
        why={
            "path": "A path starts at the course folder and names each folder, with `/` between them, down to the file.",
            "name": "A path starts at the course folder, and the file is not there. See the table of new words in [Reading a text file](/course/open/py-files?lang=en).",
            "back": "A path goes from the outside in: first the folder you start in, the file last. See the recipe in [Reading a text file](/course/open/py-files?lang=en).",
        },
    )
    quick_path
    return (quick_path,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Not all data fits one grid of numbers of one type. In a table of houses, each house has an address (text), a price and a number of rooms, and each column has a name; an array has one type and no names. pandas, a library for tables, keeps such data as a DataFrame: a name and a type for each column, and a row per example. Tables like this often come as a CSV file, a text file with a table in it.

    **An everyday example.** A class list in a spreadsheet: a column of names, a column of ages and a column of grades, each with its heading at the top.
    """)
    return


@app.cell(hide_code=True)
def predict_filter():
    guess_filter = L.guess(
        "filter",
        question="`df` is a table of a week of weather, with the columns `day`, `temp`, `rain` and `wind`. What do you think `df[df[\"rain\"] > 0]` gives?",
        options={
            "rows": "The rainy days, with all four columns",
            "column": "Only the column `rain`, of the rainy days",
            "mask": "`True` or `False` for every day",
        },
        reveal="**The rainy days, with all four columns.** `df[\"rain\"] > 0` is a mask, `True` for every day with rain, just like a mask on an array. Between the square brackets of `df`, it keeps those rows, whole.",
    )
    guess_filter
    return (guess_filter,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    pandas is a library for tables. Its table is a DataFrame: columns with a name, each of one type, and a row per example. You read one from a CSV file, take its columns by their names, and keep the rows you want with a mask, as with an array.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **library** | a set of modules that someone else wrote, installed with the course; NumPy is one too | |
    | **pandas** | a library for tables; it is always imported as `pd` | `import pandas as pd` |
    | **CSV file** | a text file with a table: a row per line, the values separated by commas, and the column names on the first line | `day,temp,rain,wind` |
    | **DataFrame** | the table of pandas: named columns, each of one type, and a row per example | `df = pd.read_csv(path)` |
    | **row label** | the number pandas puts to the left of each row; selected rows keep their old row labels; pandas calls them the index | `0`, `1`, `2` … |
    | **one column** | its name between the square brackets; you get a Series: the values, with their row labels | `df["temp"]` |
    | **some columns** | a list of names between the square brackets, so two pairs of brackets; you get a new DataFrame | `df[["day", "temp"]]` |
    | **selecting rows** | a mask on a column, between the square brackets of the DataFrame, keeps the rows where it is `True` | `df[df["rain"] > 0]` |

    `df.shape` gives the number of rows and columns, as with an array, and `df.head()` gives the first five rows. pandas itself is built on NumPy.
    """)
    return


@app.cell(hide_code=True)
def picture_select():
    mo.vstack([
        draw.data_tables(C.picture(["one column", "selected rows"]), title="Picking from a DataFrame",
                         alt="Picking from a DataFrame. The weather table twice: on the left, the column rain is taken; on the right, the rows of Tuesday, Wednesday, Thursday and Sunday, the days with rain, are kept whole."),
        mo.md(r"""
        The same table twice. On the left, `df["rain"]` takes one column by its name. On the right, `df[df["rain"] > 0]` keeps the rows of the four days with rain, with all their columns. The small numbers on the left are the row labels: the kept rows keep theirs.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("For more than one column, the names go in a list: two pairs of square brackets, `df[[\"day\", \"temp\"]]`. With one pair, `df[\"day\", \"temp\"]`, pandas looks for one column with that whole name, and stops with an error."),
        mo.md(r"""
        ### Recipe

        To work with a table:

        1. Import pandas and read the table: `df = pd.read_csv(path)`.
        2. Look at it: `df.shape` for its size, and `df.head()` for its first rows.
        3. Take columns by their names: `df["temp"]`, or `df[["day", "temp"]]`.
        4. Keep the rows you want with a mask on a column: `df[df["rain"] > 0]`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up week of weather, in `lessons/data-pandas/weather.csv`: for each day, the temperature in degrees Celsius, the rain in millimeters, and the wind in kilometers per hour. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_weather():
    # 1. import pandas and read the table
    import pandas as pd

    _df = pd.read_csv("lessons/data-pandas/weather.csv")
    # 2. its size, and its first five rows
    print(_df.shape)
    print(_df.head())
    # 3. one column, and two columns
    print(_df["temp"])
    print(_df[["day", "rain"]])
    # 4. the mask of the rainy days, the rows it keeps, and the temperatures of the dry days
    print(_df["rain"] > 0)
    print(_df[_df["rain"] > 0])
    print(_df[_df["rain"] == 0]["temp"])
    return (pd,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The table has 7 rows and 4 columns, `(7, 4)`, and `head` shows Monday to Friday. The column `temp` comes out as a Series: each temperature with its row label, and below it its name and dtype. Two columns come out as a smaller DataFrame. The mask is `True` or `False` for each day, and the rows it keeps are rows 1, 2, 3 and 6: selected rows keep their old row labels. The last line first keeps the dry days, and then takes their temperatures: `12.5`, `13.0` and `16.5`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_columns():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two columns
    """)
    return


@app.cell(hide_code=True)
def choice_columns_cell():
    choice_columns = L.choice(
        "two-columns",
        question="Which line gives a DataFrame with only the columns `day` and `wind`?",
        options={
            "one": "`df[\"day\", \"wind\"]`",
            "list": "`df[[\"day\", \"wind\"]]`",
            "chain": "`df[\"day\"][\"wind\"]`",
        },
        correct="list",
        why={
            "one": "With one pair of brackets, pandas looks for one column with the name `\"day\", \"wind\"`, and stops with an error. Look at the *Watch out* box above the recipe again.",
            "chain": "`df[\"day\"]` is already one column, and that column has no column `wind` in it. How do you ask for several columns at once?",
            "list": "The inner brackets make a list of two names, and the outer ones take those columns from `df`.",
        },
    )
    choice_columns
    return (choice_columns,)


@app.cell(hide_code=True)
def text_exercise_windy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The windy days

        The box reads the week of weather as `df`. Make `windy`: the rows of the days with more wind than 20 kilometers per hour, with all their columns. Replace the `...`. The last line is `windy`, so the check sees it. The check also tries your code on another week.
        """),
        L.task("Make `windy`: the rows of `df` where the wind is more than 20."),
    ])
    return


@app.cell(hide_code=True)
def exercise_windy():
    box_windy = L.code_box("windy", lines=6)
    box_windy
    return (box_windy,)


@app.cell(hide_code=True)
def check_windy(box_windy):
    result_windy = L.check("windy", box_windy.value)
    result_windy
    return (result_windy,)


@app.cell(hide_code=True)
def hints_windy():
    L.hints(
        "windy",
        [
            "Step 4 of the recipe: a mask on one column keeps rows.",
            "The mask compares the column `wind` with 20. Then it goes between the square brackets of `df`.",
            "`windy = df[___]`, with the mask between the brackets.",
        ],
        why="`df[\"wind\"] > 20` is `True` for Tuesday, Wednesday and Sunday. Between the square brackets of `df`, that mask keeps those three rows, with all four columns.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_warm():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The warm days of any week

        Finish `warm_days(path, limit)`. It reads the CSV file at `path`, a week of weather with the same four columns, as `df`. It gives back the names of the days that are warmer than `limit`: the column `day` of those rows. Replace the `...`. The check calls `warm_days("lessons/data-pandas/weather.csv", 13)`: that should give the days `tue`, `sat` and `sun`. It also tries another week, in `lessons/data-pandas/weather-2.csv`.
        """),
        L.task("Make `warm_days(path, limit)` give back the names of the days warmer than `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_warm():
    box_warm = L.code_box("warm-days", lines=8)
    box_warm
    return (box_warm,)


@app.cell(hide_code=True)
def check_warm(box_warm):
    result_warm = L.check("warm-days", box_warm.value)
    result_warm
    return (result_warm,)


@app.cell(hide_code=True)
def hints_warm():
    L.hints(
        "warm-days",
        [
            "Two steps: first keep the warm rows, with a mask; then take one column of them.",
            "The mask compares the column `temp` with `limit`. Of the rows it keeps, you want the column `day`.",
            "`warm = df[___]`, and then `return warm[___]`.",
        ],
        why="`df[df[\"temp\"] > limit]` keeps the rows warmer than `limit`, and `[\"day\"]` takes their names. The function reads the file at `path`, so it works for any week in the same form.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Selecting parts of an array](/course/open/np-indexing?lang=en) a mask picked numbers from an array; in a DataFrame, a mask on one column selects whole rows. In [Reading a text file](/course/open/py-files?lang=en) you read a file line by line; `pd.read_csv` reads a whole table at once, with its column names. In the next lesson, *Describing data: mean, median and spread*, you summarize a column with a few numbers and a picture.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_columns, result_warm, result_windy):
    L.wrap_up(choice_columns, result_windy, result_warm)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
