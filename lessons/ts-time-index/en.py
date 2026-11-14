# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ts-time-index. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-time-index", "en")
    C = L.checks   # the traffic of every hour with the time as its index, the rolling means of the slider, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en) and one about [One number per group: groupby](/course/open/data-groupby?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hours():
    quick_hours = L.choice(
        "quick-hours",
        question="The traffic table has 48,204 rows but only 40,575 different hours in its column `date_time`. Which line leaves one row for every hour?",
        options={
            "all": "`df.drop_duplicates()`",
            "right": "`df.drop_duplicates(subset=\"date_time\")`",
            "na": "`df.dropna()`",
        },
        correct="right",
        why={
            "all": "Without a column, `drop_duplicates` only drops rows that are the same in every column: 17 here. Most repeated hours differ in their weather. See *The idea* in [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en).",
            "na": "`dropna` drops rows with a missing value; a repeated hour has none. See *The idea* in [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en).",
            "right": "One row per hour, the first of each. Today those hours become the index of the table.",
        },
    )
    quick_hours
    return (quick_hours,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question="`df` has a column `weekday`, 0 for Monday to 6 for Sunday, and a column `traffic_volume`. What does `df.groupby(\"weekday\")[\"traffic_volume\"].mean()` give?",
        options={
            "right": "Seven numbers: the mean traffic of each weekday",
            "rows": "A number for every row: the mean of its weekday",
            "one": "One number: the mean of all traffic",
        },
        correct="right",
        why={
            "rows": "`groupby` gives one number per group, not one per row: seven groups, seven numbers. See *The idea* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "one": "That is `df[\"traffic_volume\"].mean()`. `groupby(\"weekday\")` first splits the rows into seven groups. See *The idea* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "right": "One number per group. Today the groups are stretches of time, such as days, and pandas makes them for you.",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The traffic table of your project counts the vehicles on Interstate 94 every hour, from October 2012 to September 2018. What does a usual day look like? Is Friday busier than Monday? Does the traffic grow over the years? Every one of those questions is about time: cut out a month, add the hours up to days, smooth away the bumps of single hours, compare an hour with the same hour a week earlier. pandas does all of that, once the time is the index of the table.

    **An everyday example.** A step counter shows your steps per hour, per day and per week. On a day you leave your phone at home, it shows 0 steps. You did walk; nothing counted them. A time series has the same trap.
    """)
    return


@app.cell(hide_code=True)
def predict_zero_days():
    guess_zero = L.guess(
        "zero-days-guess",
        question="With the time as the index, `s.resample(\"D\").sum()` adds up the hours of every day from 2 October 2012 to 30 September 2018: 2,190 days. On how many of them is the total 0?",
        options={
            "none": "None: a highway is never empty for a whole day",
            "few": "About ten, on holidays and in snowstorms",
            "right": "About 330",
        },
        reveal="**330.** On 330 days the table has no row at all, and the sum of nothing is 0. A day nobody counted looks like a day without a single vehicle. `s.resample(\"D\").mean()` gives NaN for those days instead, and `s.resample(\"D\").count()` says how many hours each day rests on: 1,214 days have all 24, and 646 have only some.",
    )
    guess_zero
    return (guess_zero,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Time as the index.** A **time series** is a number measured one moment after another, such as the traffic of every hour. In pandas, put the times in the index: `s = df.set_index("date_time")["traffic_volume"]` is a Series whose index is a **datetime index**, one moment per value. pandas then reads dates as you write them: `s.loc["2017-10"]` is October 2017, 744 hours, and `s.loc["2017-10-02":"2017-10-08"]` is the week from Monday 2 to Sunday 8 October, both days included.

        **Every hour in its place.** `s.asfreq("h")` gives every hour from the first to the last, NaN where the table has no row: 52,551 hours, 11,976 of them NaN. A missing hour now shows up instead of hiding.

        **Resampling.** `s.resample("D")` puts the values into bins of a day, as `groupby` puts rows into groups; `.mean()`, `.sum()` or `.count()` then gives one number per day. `"W"` makes weeks. That is **resampling**.

        **Smoothing.** A **rolling window** gives, for every hour, a number over the hours just before it: `s.rolling("24h").mean()` is the mean of the last 24 hours, that hour included. It smooths away the bumps of single hours, and it runs behind: it only knows the past. An **exponential moving average** keeps one level and moves it a step towards every new value: new level = α × new value + (1 − α) × old level. With α = 0.5 the newest value weighs 0.5, the one before 0.25, the one before that 0.125: old values fade but never quite vanish. In pandas: `s.ewm(alpha=0.3, adjust=False).mean()`.

        **Shifting.** `s.shift(1)` moves every value one row later, so each row sees the row before it. On `s.asfreq("h")`, where every row is an hour, `shift(24)` is the same hour a day earlier.

        **Time zones.** The times in the table are local times in Minneapolis. Twice a year the clocks jump. On the night of 12 March 2017 they went from 1:59 straight to 3:00: the table has no row at 2:00, and `asfreq("h")` puts a NaN there, at an hour that never was. On 6 November 2016 they went from 1:59 back to 1:00: 1:00 happened twice, and the table has one row for it. `s.tz_localize("America/Chicago")` tells pandas the **time zone**; on this table it stops with an error, *Cannot infer dst time from 2012-11-04 01:00:00*: on that night too, 1:00 happened twice, and pandas cannot tell which of the two the row is. Times in UTC, the time of the zero meridian, never jump; `tz_convert("UTC")` turns localized times into it.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **time series** | a number measured one moment after another, such as the traffic of every hour | |
        | **datetime index** | an index of moments in time, so pandas can cut, bin and shift by time | `df.set_index("date_time")` |
        | `.loc` with a date | the rows of that year, month, day or stretch, both ends included | `s.loc["2017-10"]` |
        | `asfreq` | every step from the first to the last, NaN where there was no row | `s.asfreq("h")` |
        | **resampling** | putting the values into bins of time, such as days, and giving one number per bin | `s.resample("D").mean()` |
        | **rolling window** | the stretch of time just before every moment, here to take a mean over | `s.rolling("24h").mean()` |
        | **exponential moving average** | a level that moves a share α towards every new value | `s.ewm(alpha=0.3, adjust=False).mean()` |
        | α | the share of the new value in the new level, between 0 and 1: larger follows faster | `alpha` |
        | `shift` | every value moved that many rows later | `s.shift(24)` |
        | **time zone** | the rule that turns a moment into local clock time, jumps included | `s.tz_localize("America/Chicago")` |
        """),
        L.watch_out("A window of 24 rows is not a day. `s.rolling(24)` takes the last 24 rows, and where hours have no row, those rows reach further back: for 11,503 of the 40,575 hours, the last 24 rows span more than a day, up to 320 days. `s.rolling(\"24h\")` takes the hours of the last 24 hours, however many have a row. The same holds for `shift`: on the table itself, `shift(24)` is 24 rows back; only on `s.asfreq(\"h\")` is it a day."),
        mo.md(r"""
        The picture shows four whole weeks of traffic, from Monday 2 to Sunday 29 October 2017, without a missing hour. The thin gray line is every hour; the thick line is `s.rolling(...).mean()` over the window you choose. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_window_cell():
    def _frame(hours):
        said = C.said(hours, L.lang)
        alt = ("A line chart of the traffic on Interstate 94 every hour from Monday 2 to Sunday 29 October 2017, between 0 and "
               "7,500 vehicles, with a tick at every Monday, and the rolling mean over the chosen window as a thicker line. " + said)
        return (C.picture(hours, {"title": "Traffic, 2 to 29 October 2017", "ylabel": "vehicles per hour",
                                  "days": ["Mon 2 Oct", "Mon 9 Oct", "Mon 16 Oct", "Mon 23 Oct"], "alt": alt}), said)

    explore_window = L.explore("window", "window (hours)", _frame, start=24, values=C.WINDOWS, role="hyper",
                               task="Go from a window of 1 hour to 168. At which window is the shape of a day gone, and what is left?")
    explore_window
    return


@app.cell(hide_code=True)
def text_after_window():
    mo.md(r"""
    Each window smooths away what is shorter than itself. 3 hours keep the morning and the afternoon peak, lower and an hour later. 24 hours, a whole day, leave no trace of the day: what is left is the week, low on Saturday and Sunday. 168 hours, a whole week, leave almost nothing of the week either. Which window you need depends on your question: the shape of a day needs hours, a trend over the years needs weeks or more.

    ### Recipe

    To work with a time series in pandas:

    1. Make the time a time with `pd.to_datetime`, and put it in the index with `set_index`.
    2. Put every step in its place with `asfreq`, such as `s.asfreq("h")`: a missing hour becomes NaN, so you see it.
    3. Resample to the step your question needs, with a mean rather than a sum, and count how many values each bin rests on.
    4. Smooth with a window of time, `rolling("24h")`, or with `ewm`, and compare with earlier with `shift` on the grid of `asfreq`.
    5. Know the time zone of your times: local times jump twice a year.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes the series, cuts out October 2017, puts every hour in its place, resamples to days and counts their hours, and compares Monday 2 October at 8:00 with the same hour a day earlier. Last, it prints the mean of the last 24 hours and the exponential moving average with α = 0.3, both at Monday 23:00. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_series():
    import pandas as pd

    # 1. the time as the index, one row per hour
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    _df = _df.drop_duplicates(subset="date_time")
    _df["date_time"] = pd.to_datetime(_df["date_time"])
    _s = _df.set_index("date_time")["traffic_volume"]
    print(len(_s.loc["2017-10"]))

    # 2. every hour in its place
    _hours = _s.asfreq("h")
    print(len(_hours), _hours.isna().sum())

    # 3. per day: the mean, and how many hours it rests on
    _days = _s.resample("D").mean()
    _counts = _s.resample("D").count()
    print(len(_days), _days.isna().sum(), (_counts == 24).sum())

    # 4. a day earlier, and two ways to smooth
    _day_ago = _hours.shift(24)
    print(_hours.loc["2017-10-02 08:00"], _day_ago.loc["2017-10-02 08:00"])
    print(round(_s.rolling("24h").mean().loc["2017-10-02 23:00"]), round(_s.ewm(alpha=0.3, adjust=False).mean().loc["2017-10-02 23:00"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    October 2017 has all its 744 hours. Of the 52,551 hours from the first to the last, 11,976 are NaN; of the 2,190 days, 330 have no hour at all and 1,214 have all 24. At 8:00 on Monday 2 October, 6,054 vehicles passed; a day earlier, on Sunday, 2,403. The last 24 hours up to Monday 23:00 are Monday itself, so their mean, 3,443, is the mean of that day. The exponential moving average is 2,098: it weighs the quiet late evening most.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_week_change():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A week earlier

        `week_change(s)` should give, for every hour, how many more vehicles passed than in the same hour a week earlier: NaN where either hour has no row, and for the first week, which has no week before it. `s` is a Series of hours with the time as its index. Replace the two `...`. The check calls `week_change(may)`, the four weeks from Monday 2 May 2016, 672 hours of which 654 have a row, and tries three more: two whole weeks of October 2017, four days, and ten made-up days with four hours missing.
        """),
        L.task("Let `week_change` give the change from the same hour a week earlier."),
    ])
    return


@app.cell(hide_code=True)
def exercise_week_change():
    box_week_change = L.code_box("week-change", lines=4)
    box_week_change
    return (box_week_change,)


@app.cell(hide_code=True)
def check_week_change(box_week_change):
    result_week_change = L.check("week-change", box_week_change.value)
    result_week_change
    return (result_week_change,)


@app.cell(hide_code=True)
def hints_week_change():
    L.hints(
        "week-change",
        [
            "First make sure that one row is one hour, also where the table has no row. Then subtract, from every hour, the hour a week before it.",
            "*Every hour in its place* in *The idea* says how to get every hour, NaN where there is no row. A week has 7 × 24 hours.",
            "`hours = s.asfreq(___)` and `return hours - hours.shift(___)`.",
        ],
        why="In May 2016, 18 of the 672 hours have no row. On the grid of `asfreq(\"h\")`, 168 rows back is exactly a week back; on the table itself, it is not, wherever an hour is missing in between.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_smooth():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. An exponential moving average by hand

        `smooth(values, alpha)` should give the exponential moving average after every value of the list `values`, as a list: the level starts at the first value, and moves a share `alpha` towards every value after it. Replace the `...` with the new level. The check calls `smooth([10, 20, 20, 0], 0.4)`, and tries four more: the 24 hours of Monday 2 October 2017 with `alpha` 0.3, a single value, `alpha` 1, and a level that falls to 0. It compares with what pandas gives with `ewm(alpha=alpha, adjust=False).mean()`.
        """),
        L.task("Let `smooth` move the level a share `alpha` towards every new value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_smooth():
    box_smooth = L.code_box("smooth", lines=7)
    box_smooth
    return (box_smooth,)


@app.cell(hide_code=True)
def check_smooth(box_smooth):
    result_smooth = L.check("smooth", box_smooth.value)
    result_smooth
    return (result_smooth,)


@app.cell(hide_code=True)
def hints_smooth():
    L.hints(
        "smooth",
        [
            "The new level is made of two parts: some of the new value, and the rest of the old level. `alpha` says how much of the new value.",
            "In the loop, the new value is `x` and the old level is `level`. *Smoothing* in *The idea* has the rule in words.",
            "`level = alpha * ___ + (1 - alpha) * ___`.",
        ],
        why="The level starts at 10. Then 0.4 · 20 + 0.6 · 10 = 14, 0.4 · 20 + 0.6 · 14 = 16.4, and 0.4 · 0 + 0.6 · 16.4 = 9.84: `[10, 14, 16.4, 9.84]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spring():
    mo.md(r"""
    ### Exercise 3. An hour that never was
    """)
    return


@app.cell(hide_code=True)
def choice_spring_cell():
    choice_spring = L.choice(
        "spring",
        question="On the night of 12 March 2017, the table goes from 1:00 straight to 3:00, and `s.asfreq(\"h\")` puts a NaN at 2:00. How much traffic did the station miss at 2:00?",
        options={
            "right": "None: that night the clocks went from 1:59 straight to 3:00, so 2:00 never happened in Minneapolis",
            "sensor": "An hour of traffic: the station missed it, like the other single hours without a row",
            "duplicate": "None: the row was dropped because it repeated an hour",
        },
        correct="right",
        why={
            "sensor": "Most missing hours are hours nobody counted, but not this one: on the second Sunday of March, the clocks in Minneapolis jump from 1:59 to 3:00. In local time, 2:00 does not exist that night, and `asfreq(\"h\")` cannot know that. See *Time zones* in *The idea*.",
            "duplicate": "Keeping the first row of every hour drops a second row of the same hour; it leaves the first. There was never a row at 2:00, because that night the clocks jumped from 1:59 to 3:00. See *Time zones* in *The idea*.",
            "right": "In local time, 2:00 did not exist that night. `asfreq(\"h\")` counts hours on a clock without jumps, so it adds an hour that never was. In UTC the same night has no gap.",
        },
    )
    choice_spring
    return (choice_spring,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Resampling is the `groupby` of [One number per group: groupby](/course/open/data-groupby?lang=en) with groups made of time, and `asfreq` shows as NaN the 11,976 hours you counted in [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en). The rolling mean over one season comes back in the next lesson, *Trend, season and the rest*, where it finds the trend of a series; `shift` comes back when a series is compared with itself a step earlier.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spring, result_smooth, result_week_change):
    L.wrap_up(result_week_change, result_smooth, choice_spring)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
