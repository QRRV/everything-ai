# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-project-table-explore. Route: course/route/lessen/klassiek-ml.yaml. Checks, the table and the play block: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-project-table-explore", "en")
    C = L.checks   # the traffic table, its facts, the calendar of the play block, and the demo

    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def names_in(names):
        """Names in words: September, October and November."""
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Asking questions of data](/course/open/data-explore?lang=en) and [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_describe():
    quick_describe = L.choice(
        "quick-describe",
        question="`df.describe()` of a year of weather says, for the column `temp` in degrees Celsius: `count` 365, `mean` 11.2, `min` -999.0, `max` 34.1. What does the `min` tell you?",
        options={
            "cold": "That one day was very cold",
            "right": "That a value cannot be right: a code or a mistake, to look at before anything else",
            "fine": "Nothing: `min` and `max` only show the range",
        },
        correct="right",
        why={
            "cold": "No place people measure gets near -999 °C: it is far below the coldest anything can be, -273.15 °C. A value like that is a code for *not measured*, or a mistake. See [Asking questions of data](/course/open/data-explore?lang=en).",
            "fine": "The range is exactly where a value that cannot be shows up first. -999 °C is far below the coldest anything can be, so it is a code or a mistake. See [Asking questions of data](/course/open/data-explore?lang=en).",
            "right": "`min` and `max` are where impossible values show up first. Today, `describe()` of a real table shows two of them, and you write a rule for each.",
        },
    )
    quick_describe
    return (quick_describe,)


@app.cell(hide_code=True)
def quick_check_fill():
    quick_fill = L.choice(
        "quick-fill",
        question="A column of temperatures has gaps, and you fill them in with the mean of the column. What happens to its mean and its standard deviation?",
        options={
            "both": "Both stay the same",
            "right": "The mean stays the same; the standard deviation gets smaller",
            "mean": "The mean changes; the standard deviation stays the same",
        },
        correct="right",
        why={
            "both": "The mean stays the same, but every filled-in value lies exactly on the mean, so the values spread less around it, and the standard deviation gets smaller. See [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en).",
            "mean": "Adding values that are equal to the mean leaves the mean as it was. What changes is the spread: every new value lies on the mean, so the standard deviation gets smaller. See [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en).",
            "right": "Filling in changes the numbers, even when it keeps the mean. Today you meet gaps of another kind: whole hours without a row, where there is no cell to fill.",
        },
    )
    quick_fill
    return (quick_fill,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A model learns from the table you give it, mistakes and all: it cannot tell a measurement from a typing error. So before any model, you look at the table. This step starts your project for classical machine learning: one real table, which you take from a first look to a model you can trust, step by step. The table is the traffic on Interstate 94, a highway in Minnesota, heading west, counted every hour by station 301 of the state's Department of Transportation, roughly midway between Minneapolis and Saint Paul, from October 2012 to September 2018. Each hour comes with the weather of that hour, and the public holidays. John Hogue put the table together and shared it in 2019 on the UCI Machine Learning Repository, under the license CC BY 4.0, so the course ships it, with his name.

        Today, you look at it. You will find temperatures of 0 K, which nothing ever reaches, an hour with more rain than ever fell anywhere, hours that are in the table more than once, and months without a single row. For each problem, you write a rule, and you keep the rules in your project: every later step starts from the same clean table.

        **An everyday example.** You count the visitors of a museum from its ticket list. Some tickets were scanned twice, the scanner was broken for a week in winter, and one day says 90,000 visitors. You would not take the average of that list as it is: first you fix what can be fixed, and you write down what cannot.
        """),
        draw.project_steps(["Your table, explored and cleaned", "Trees on your table, fairly"], 0,
                           here_label="← today", title="Where your project stands: your table"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_repeats():
    guess_repeats = L.guess(
        "repeats",
        question="The table has 48,204 rows, one for every hour, you would think, from 2 October 2012 to 30 September 2018. How many of those rows repeat an hour that already has a row?",
        options={
            "none": "None: every hour is in it once",
            "few": "About 20",
            "right": "About 7,600",
        },
        reveal="**7,629**, spread over 5,445 hours; two hours even have 6 rows. Only 17 of them are the same as another row in every column. Most repeat an hour with another kind of weather: at 7:00 on 10 October 2012, one row says *Rain* and another *Drizzle*, with the same temperature and the same traffic.",
    )
    guess_repeats
    return (guess_repeats,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The table.** It has 9 columns: `holiday`, the name of a holiday, only in the first hour of that day; `temp`, the temperature in kelvin; `rain_1h` and `snow_1h`, the rain and the snow in that hour, in mm; `clouds_all`, the share of the sky covered by clouds, in %; `weather_main` and `weather_description`, the weather in a word and in a few; `date_time`, the hour, in local time; and `traffic_volume`, how much traffic the station counted heading west in that hour. Kelvin is a temperature scale that starts at the coldest anything can be: 0 K is -273.15 °C, and degrees Celsius are kelvin minus 273.15.

        **Impossible values.** `df.describe()` shows them in its `min` and `max`. The lowest temperature is 0.00 K: 10 hours, between 03:00 and 08:00, on 31 January and 2 February 2014. The lowest after those is 243.39 K, -29.76 °C, a cold December morning in Minnesota. The most rain in an hour is 9,831.3 mm, at 17:00 on 11 July 2016: almost 10 meters of water in one hour, taller than a house. The most after that is 55.63 mm. Neither can be true, so both go, with a rule for each: a mask that keeps the rows that can be, such as `df[df["temp"] > 0]`. Choose the limit from what is possible, not from the one row you saw: a limit of 100 mm of rain in an hour is far above anything this station measured, the 55.63 mm, and far below the 9,831.3 that cannot be. A rule written that way also works on hours the table does not have yet.

        **Duplicate rows.** A table of hours should have each hour once. This one has 48,204 rows but only 40,575 different hours: 7,629 rows repeat an hour. `df.duplicated()` finds only 17 of them, because it looks for rows that are the same in every column. Most repeated hours differ in their weather: the station reported two kinds at once, and each got its own row. Name the column that should be unique, and `df.duplicated(subset="date_time")` finds all 7,629. The rule of this lesson keeps the first row of every hour, `df.drop_duplicates(subset="date_time")`: then the second kind of weather of an hour is gone, which is the price of one row per hour.

        **Missing hours.** The table's page on the UCI repository says *Has Missing Values? No*. Of its cells, that is true: `df.isna().sum()` gives 0 for every column but `holiday`, where pandas reads the word `None`, no holiday that day, as missing. But from the first hour to the last are 52,551 hours, and only 40,575 of them have a row: 11,976 hours have no row at all. `isna()` cannot find them, because there is no cell to be missing. You find them by counting the hours from the first to the last, which needs `date_time` as moments in time rather than as text: `pd.to_datetime(df["date_time"])`. The longest gap is 307 days, from 8 August 2014 at 01:00 to 11 June 2015 at 20:00. No rule brings those hours back. Imputation fills in a cell of a row that is there; filling in 307 days would mean making up thousands of hours of traffic that nobody counted.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **data cleaning** | fixing the problems of raw data before you use it, such as impossible values, duplicate rows and wrong types | |
        | **duplicate row** | a row that repeats what should be in the table once, such as a second row for the same hour | `df.duplicated(subset="date_time")` |
        | `drop_duplicates` | the table without its duplicate rows: of the rows with the same value in the column `subset`, only the first stays | `df.drop_duplicates(subset="date_time")` |
        | `pd.to_datetime` | a column of text such as `"2012-10-02 09:00:00"` as moments in time, which you can compare and count with | `pd.to_datetime(df["date_time"])` |
        | `pd.date_range` | every hour from `first` to `last`, the two included | `pd.date_range(first, last, freq="h")` |
        | **kelvin**, K | a temperature scale that starts at the coldest anything can be: 0 K is -273.15 °C | `df["temp"] - 273.15` |
        """),
        L.watch_out("`df.duplicated()` and `df.drop_duplicates()` without a column only look for rows that are the same in every column: here 17. Two rows of the same hour with another kind of weather are not the same, so name the column that should be unique: `subset=\"date_time\"`."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    One year of the table as a calendar: a column per month, a row per day. The fuller a day, the more of its 24 hours have a row; a dashed day has none. A dot marks a day with an hour that has more than one row. Slide through the years, from 2012 to 2018.
    """)
    return


@app.cell(hide_code=True)
def explore_year_cell():
    _legend = ["all 24 hours have a row", "some hours have no row", "no row at all", "an hour with more than one row"]

    def _frame(y):
        y = int(y)
        _f = C.year_facts(y)
        _since = {2012: " (from 2 October, 09:00)", 2018: " (to 30 September)"}.get(y, "")
        _gone = [MONTHS[m - 1] for m in _f["months_gone"]]
        _no = f" No row at all in {names_in(_gone)}." if _gone else " Every month has rows."
        _said = (f"**{y}**{_since}: {C.grouped(_f['have'], L.lang)} of its {C.grouped(_f['hours'], L.lang)} hours have a row, "
                 f"and {C.grouped(_f['repeats'], L.lang)} rows repeat an hour.{_no}")
        _alt = (f"A calendar of {y}: a column per month and a row per day, each day filled by the share of its hours that "
                f"have a row, with a dot on the days with an hour that has more than one row. " + _said.replace("**", ""))
        return C.calendar(y, SHORT, _legend, f"{y}", _alt).text, _said

    explore_year = L.explore("year", "year", _frame, start=2015, values=C.YEARS, role="",
                             task="Find the hole of 306 empty days in a row. In which years is no day empty?")
    explore_year
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    The big hole runs from August 2014 to June 2015; 63 of the 72 months have rows. Around it, shorter gaps remain, from one missing hour to ten days in 2013: 2,588 jumps of more than one hour in all. And the dots are everywhere: every year has hundreds of repeated hours, most of them with mist, rain or snow, when the station reported more than one kind of weather.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Before any model, for every new table:

    1. Read it and look: `df.describe()` for values that cannot be, and `len(df)` against the number of different values in the column that should be unique.
    2. For every impossible value, write a rule: a mask that keeps the rows that can be, such as `df = df[df["temp"] > 0]`, with a limit chosen from what is possible.
    3. One row for every hour: `df.drop_duplicates(subset="date_time")` keeps the first row of each.
    4. Make text that is a time a time: `pd.to_datetime(df["date_time"])`, and put a measurement in a unit you can read, such as degrees Celsius instead of kelvin. Count the hours from the first to the last, and compare that with the hours that have a row.
    5. Count what each rule leaves, and write down what no rule can fix, such as the missing hours, in the datasheet of the table.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads the table, prints the numbers of `describe()` for three columns, counts the impossible values and the repeated hours, and counts the hours from the first to the last. Run the cell.
    """)
    return


@app.cell
def example_look():
    import pandas as pd

    # 1. read the table, and look at three of its columns
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    print(len(_df))
    print(_df[["temp", "rain_1h", "traffic_volume"]].describe())

    # 2. the impossible values: 0 K, and more than 100 mm of rain in an hour
    print((_df["temp"] == 0).sum(), (_df["rain_1h"] > 100).sum())

    # 3. repeated rows: the same in every column, and the same hour
    print(_df.duplicated().sum(), _df.duplicated(subset="date_time").sum())
    print(_df[_df["date_time"] == "2012-10-10 07:00:00"])

    # 4. the hours from the first to the last, and the hours that have a row
    _times = pd.to_datetime(_df["date_time"])
    _every = pd.date_range(_times.min(), _times.max(), freq="h")
    print(_times.min(), _times.max())
    print(len(_every), len(_times.drop_duplicates()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `describe()` shows `min` `0.000000` for `temp`, and `max` `9831.300000` for `rain_1h`: 10 hours at 0 K and 1 hour with more than 100 mm of rain. 17 rows are the same as another row in every column, and 7,629 repeat an hour. The two rows of 10 October 2012 at 7:00 differ only in their weather. From the first hour to the last are 52,551 hours; 40,575 of them have a row.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_impossible():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A rule for each impossible value

        `drop_impossible(df)` should leave out the rows that cannot be right: a temperature of 0 K, and the rain of 9,831.3 mm. As the rule for rain, keep the hours with at most 100 mm, far above anything this station measured. Replace the two `...` with a line each, of the form `df = df[...]`. The check calls `drop_impossible(TRAFFIC)`, with the whole table as `read_csv` gives it, and tries four small tables of its own, one without a single wrong row. When it is right, the check saves it in your project, in `work/projects/tabular/impossible.py`.
        """),
        L.task("Keep the rows whose `temp` is above 0 and whose `rain_1h` is at most 100."),
    ])
    return


@app.cell(hide_code=True)
def exercise_impossible():
    box_impossible = L.code_box("impossible", lines=6)
    box_impossible
    return (box_impossible,)


@app.cell(hide_code=True)
def check_impossible(box_impossible):
    result_impossible = L.check("impossible", box_impossible.value)
    result_impossible
    return (result_impossible,)


@app.cell(hide_code=True)
def hints_impossible():
    L.hints(
        "impossible",
        [
            "Step 2 of the recipe. A mask in `df[...]` keeps the rows where it is True: which rows should stay?",
            "`df[\"temp\"] > 0` is True for every temperature that can be; for the rain, it is `<=` and 100.",
            "`df = df[df[\"temp\"] ___ 0]` and `df = df[df[\"rain_1h\"] ___ 100]`.",
        ],
        why="Each line keeps the rows that pass its rule, and the next line works on what is left. Because the limits are rules and not row numbers, they work on the small tables of the check too, and on hours the table does not have yet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One row for every hour

        `one_per_hour(df)` should give back the table with one row for every hour: of the rows with the same `date_time`, only the first. Replace the `...`. The check calls `one_per_hour(TRAFFIC)` and two small tables, one without any hour twice, and saves your function in `work/projects/tabular/hours.py`.
        """),
        L.task("Leave out every row whose `date_time` is already in the table, and keep the first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("hours", lines=4)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("hours", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "hours",
        [
            "Step 3 of the recipe. Which column should be unique, and how do you tell `drop_duplicates` to look only at that one?",
            "`subset=\"date_time\"` makes `drop_duplicates` compare only the hour; it keeps the first row by itself.",
            "`return df.drop_duplicates(subset=___)`",
        ],
        why="Without `subset`, only rows that are the same in every column go: 17. With it, every second, third or later row of an hour goes: 7,629.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. clean: your rules together

        `clean(df)` takes your two rules from your project, `from tabular.impossible import drop_impossible`, and then makes two columns easier to use: `date_time` as moments in time, and `temp` in degrees Celsius. Replace the two `...`. The check calls `clean(TRAFFIC)` and four small tables. Where a module of yours is missing or broken, the course's version stands in, as the note under *Where your project stands* says. When it is right, the check saves it in `work/projects/tabular/clean.py`: the next step of your project starts from it.
        """),
        L.task("Make `date_time` a column of moments in time, and `temp` degrees Celsius."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clean():
    box_clean = L.code_box("clean", lines=12)
    box_clean
    return (box_clean,)


@app.cell(hide_code=True)
def check_clean(box_clean):
    result_clean = L.check("clean", box_clean.value)
    result_clean
    return (result_clean,)


@app.cell(hide_code=True)
def hints_clean():
    L.hints(
        "clean",
        [
            "Step 4 of the recipe, after your rules. `df[\"date_time\"] = ...` puts a new version in the place of the column: what makes text a time, and how do you get degrees Celsius from kelvin?",
            "`pd.to_datetime` takes one column; degrees Celsius are kelvin minus 273.15.",
            "`df[\"date_time\"] = pd.to_datetime(df[___])` and `df[\"temp\"] = df[\"temp\"] ___ 273.15`.",
        ],
        why="Your rules run first: the rule for 0 K needs the temperature in kelvin, because after the change it would be -273.15. Then `pd.to_datetime` turns the text into moments in time, and the temperature gets the scale of a weather report.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell takes your `clean` from your project, runs it on the whole table, and counts what each step leaves. From your clean table, it draws the mean traffic for every hour of the day, on working days and at weekends.
    """)
    return


@app.cell(hide_code=True)
def demo_cell(result_clean):
    _r = result_clean   # so that this cell runs again after every check of exercise 3
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercise 3 first: then your `clean` is in your project, and this cell uses it.*")
    elif isinstance(_result, str):
        _out = mo.md("*Your `clean` in `work/projects/tabular/clean.py` stops with an error, or does not give back the table with `date_time` as moments in time. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*")
    else:
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
        _first = "" if _result["first"] is None else f" your `drop_impossible` leaves **{C.grouped(_result['first'], L.lang)}**, and"
        _twice = "no hour twice" if _result["repeats"] == 0 else f"{C.grouped(_result['repeats'], L.lang)} rows that still repeat an hour"
        _low, _high = _result["temp"]
        _labels = {"title": "Mean traffic per hour of the day", "x": "hour of the day", "y": "traffic per hour",
                   "work": "working days", "weekend": "weekends",
                   "alt": "The mean traffic for every hour of the day in your clean table: on working days solid, with peaks at 7:00 and at 16:00, and at weekends dashed, with one broad hill around midday."}
        _out = mo.vstack([
            mo.md(f"Of the {C.grouped(_result['rows'], L.lang)} rows,{_first} your `clean` leaves **{C.grouped(_result['clean'], L.lang)}**: {_twice}, with temperatures from {C.number(_low, 1, L.lang)} to {C.number(_high, 1, L.lang)} °C. Your clean table still jumps more than an hour in {C.grouped(_result['gaps'], L.lang)} places: no rule can fill those.{_who}"),
            mo.Html(C.hours_chart(_result["by_hour"], _labels, L.lang)),
            mo.md("On working days, the traffic has two peaks, the morning and the afternoon rush; at weekends, one broad hill around midday. That shape is what the models of the later steps of your project will have to learn."),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.md(r"""
    ### Exercise 4. The hole of ten months
    """)
    return


@app.cell(hide_code=True)
def choice_gap_cell():
    choice_gap = L.choice(
        "gap",
        question="From 8 August 2014 at 01:00 to 11 June 2015 at 20:00, the table has no rows: almost 308 days. What do you do with that hole?",
        options={
            "fill": "Fill it in: give every missing hour the mean traffic of that hour of the day",
            "drop": "Leave out 2014 and 2015 entirely, so that no year has a hole",
            "right": "Leave it as it is: make up no hours, and write the hole down in the datasheet, so that later steps know it is there",
        },
        correct="right",
        why={
            "fill": "That makes up 7,386 hours of traffic that nobody counted, and a model would learn from them as if they were measurements. Imputation fills in a cell of a row that is there, with a missing indicator to tell the guess from a measurement; here, whole rows are missing.",
            "drop": "That throws away 8,094 hours that were counted, and the hole is still there, now between 2013 and 2016. A hole is not a mistake in the rows you have: those stay.",
            "right": "The rows you have stay as they are, and nothing is made up. Written down in the datasheet, the hole is known to every later step: when a step looks at hours in a row, it knows that the hour before 11 June 2015 at 20:00 is almost 308 days earlier.",
        },
    )
    choice_gap
    return (choice_gap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** `describe()` of [Asking questions of data](/course/open/data-explore?lang=en) showed the impossible values; the gaps of [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en) were cells, and today's were whole rows, which no `fillna` reaches. The datasheet of [Where does this data come from?](/course/open/data-provenance?lang=en) is where the hole of 307 days belongs. Your rules live in a package of your own, as in [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en): your project `tabular` now has `impossible.py`, `hours.py` and `clean.py`. The project comes back at the end of the basics of classical machine learning, in *Project: trees on your table, fairly*, where your clean table gets models. First, in the next lesson, *Learn now or later: the nearest centroid*, you meet more ways to learn from a table.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gap, result_clean, result_hours, result_impossible):
    L.wrap_up(result_impossible, result_hours, result_clean, choice_gap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
