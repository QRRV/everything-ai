# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ts-trend-season. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-trend-season", "en")
    C = L.checks   # the riders of every day, the decomposition for every window of the slider, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Time as an index: dates and resampling in pandas](/course/open/ts-time-index?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rolling():
    quick_rolling = L.choice(
        "quick-rolling",
        question="`y` has one value per day, with the date as its index. What is `y.rolling(7).mean()` on Sunday 8 January 2012?",
        options={
            "right": "The mean of the 7 days up to Sunday 8 January: Monday 2 to Sunday 8 January",
            "around": "The mean of the 3 days before Sunday 8 January, that Sunday itself and the 3 days after it",
            "week": "One number for the whole week, the same on every day of it",
        },
        correct="right",
        why={
            "around": "A rolling window ends at the day itself: it only knows the past. Today, `center=True` puts the day in the middle of its window. See *Smoothing* in *The idea* in [Time as an index](/course/open/ts-time-index?lang=en).",
            "week": "One number per week is what resampling gives, `y.resample(\"W\").mean()`. A rolling window gives a number for every day. See *Smoothing* in *The idea* in [Time as an index](/course/open/ts-time-index?lang=en).",
            "right": "The window ends at the day itself, so the mean runs behind. Today the window moves to the middle, and finds the trend of a series.",
        },
    )
    quick_rolling
    return (quick_rolling,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Capital Bikeshare rents out bikes from stations all over Washington, D.C. The table of this lesson counts the bikes rented every day of 2011 and 2012, by registered riders, members who mostly ride to work, and by casual riders with a pass for a day. How many bikes should stand ready next Monday? The numbers mix three things: the scheme grows, so 2012 is busier than 2011; every week goes up and down in the same way; and some days are a surprise, such as Monday 29 October 2012, when Hurricane Sandy closed the city and registered riders rented 20 bikes. To forecast, pull the three apart: what grows, what comes back every week, and what nobody could see coming.

    **An everyday example.** The water in a harbor rises and falls with the tide, twice a day. Over the years the sea slowly rises. On the day of a storm the water stands far higher than the tide says. Tide, rising sea, storm: a pattern that comes back, a slow movement, and a surprise.
    """)
    return


@app.cell(hide_code=True)
def predict_two_weeks():
    guess_riders = L.guess(
        "two-kinds-guess",
        question="Compared with their own trend, registered riders rent the most bikes on Thursday, 411 more, and the fewest on Sunday, 760 fewer. Casual riders rent the most on Saturday, 631 more, and the fewest on Wednesday, 300 fewer. For all riders together: how far apart are the best and the worst day of the week?",
        options={
            "add": "About 2,100: the two patterns add up",
            "right": "About 430: the two patterns mostly cancel",
            "zero": "0: the two patterns cancel exactly",
        },
        reveal="**429.** All riders together rent 171 more on Friday and 258 fewer on Sunday. Where registered riders go down, casual riders go up, and in the sum the two strong weekly patterns mostly hide each other. A series that is a sum can hide its patterns: split it where you can. This lesson follows the registered riders.",
    )
    guess_riders
    return (guess_riders,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Three parts.** Many time series are a sum of three parts: series = trend + season + remainder. The **trend** is the slow movement: registered riders grew from 2011 to 2012. The **seasonality**, or season for short, is a pattern that repeats with a fixed **period**: here a week of 7 days, so the season is 7 numbers, one per weekday. The **remainder** is what is left: weather, holidays, a hurricane. Splitting a series into these three parts is a **decomposition**.

        **The trend: a centered moving average.** For every day, take the mean of the days around it: `half` days before it, the day itself and `half` days after it, a window of `2 * half + 1` days. That is a **centered moving average**. A window of exactly one period, 7 days, holds every weekday once, so the week averages out of it, and what is left is the trend. In pandas: `y.rolling(7, center=True).mean()`; `center=True` puts the day in the middle of its window, where `rolling` puts it at the end. The first and the last 3 days get no trend, because their window does not fit. In NumPy, `np.full(n, np.nan)` makes an array of `n` values that are all `np.nan`, NumPy's NaN, to fill in day by day.

        **The season: the mean per position.** `rest = y - trend` is the series without its trend. The season of Mondays is the mean of `rest` over all Mondays, and the same for every other weekday. In NumPy, a slice with a step takes every 7th value: `rest[k::7]` is the values at positions `k`, `k + 7`, `k + 14` and so on, and `np.nanmean` is the mean that leaves NaN out, such as the days at the ends without a trend. Last, shift the seven means so that they add up to 0: the season must not move the level, which is the trend's job. In pandas, `rest.index.dayofweek` gives every date its weekday, 0 for Monday to 6 for Sunday, and `rest.groupby(rest.index.dayofweek).mean()` gives the seven means.

        **The remainder.** remainder = series − trend − season. A remainder far from 0 is a surprise worth a look. The lowest of all is Monday 29 October 2012, Hurricane Sandy: 3,799 below trend and season together.

        **A forecast.** A **forecast** is a guess of values that have not happened yet. The simplest one from a decomposition: continue the trend from its last value, and add the season of the day you forecast. The remainder is left out: it is what nobody could see coming.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **trend** | the slow movement of a series, without the pattern that repeats | `y.rolling(7, center=True).mean()` |
        | **seasonality** | a pattern that repeats with a fixed period, such as every week; the season for short | |
        | **period** | the length of the pattern that repeats: 7 days, 24 hours, 12 months | `period` |
        | **centered moving average** | for every step, the mean of a window with that step in its middle | `center=True` |
        | **remainder** | what is left of the series after the trend and the season | `y - trend - season` |
        | **decomposition** | splitting a series into trend, season and remainder | |
        | **forecast** | a guess of values that have not happened yet | |
        | `::` | a slice with a step: every 7th value from position `k` | `rest[k::7]` |
        | `np.full` | an array of the given length, with the same value everywhere | `np.full(n, np.nan)` |
        | `np.nanmean` | the mean of an array, NaN left out | `np.nanmean(rest[k::7])` |
        | `dayofweek` | the weekday of every date in a datetime index, 0 for Monday to 6 for Sunday | `y.index.dayofweek` |
        """),
        L.watch_out("A season is a mean, and a few special days pull it. Monday's season is only +15, far below Tuesday's +291: 15 of the 21 holidays in these two years fall on a Monday, and on those 15 Mondays registered riders rented 440 fewer bikes than the trend, on the other 90 Mondays 92 more. Without the holidays, Monday's season is +74. Look at what pulls a season before you trust it."),
        mo.md(r"""
        On the left, the thin gray line is the registered riders of every day of 2011 and 2012, and the thick line the trend: the centered moving average over the window you choose. On the right, the season that is left, one bar per weekday. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_window_cell():
    def _frame(days):
        said = C.said(days, L.lang)
        alt = ("Left, a line chart of the bikes rented by registered riders every day of 2011 and 2012, between 0 and 7,000, "
               "with the trend over the chosen window as a thicker line. Right, a bar chart of the season, one bar per weekday "
               "from Monday to Sunday. " + said)
        return (C.picture(days, {"left": "Registered riders and trend", "right": "Season", "ylabel": "bikes per day",
                                 "months": ["Jan 2011", "Jul 2011", "Jan 2012", "Jul 2012"], "lang": L.lang, "alt": alt}), said)

    explore_window = L.explore("trend-window", "window (days)", _frame, start=7, values=C.WINDOWS, role="hyper",
                               task="Go from a window of 3 days to 35. From which window on does the season stay the same, and what happens to the trend?")
    explore_window
    return


@app.cell(hide_code=True)
def text_after_window():
    mo.md(r"""
    A window shorter than a week leaves part of the week in the trend: with 3 days, the trend jumps 333 a day on average and the season spans only 553. From 7 days on, every window holds whole weeks, and the season stays at about 1,170 to 1,185: every such window finds the same week. A longer window gives a calmer trend, 104, 38 and then 25 a day, and leaves more of the slow movement in the remainder: its standard deviation grows from 596 to 719. One period is the shortest window that keeps the week out of the trend.

    ### Recipe

    To split a time series into trend, season and remainder:

    1. Choose the period: the length of the pattern that repeats, such as 7 days, 24 hours or 12 months.
    2. Find the trend with a centered moving average over one period.
    3. Subtract the trend, take the mean per position in the period, and shift the means so that they add up to 0: that is the season.
    4. Subtract the season too: that is the remainder. Look at its largest values.
    5. To forecast, continue the trend and add the season of the position you forecast.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell splits the registered riders in pandas. It prints the trend on 1 July of both years, the seven numbers of the season from Monday to Sunday, and the lowest remainder with its date: `idxmin` gives the index of the lowest value, as `np.argmin` gives its position. To subtract the season of every day, `_season.loc[...]` looks up the season of every weekday, and `.to_numpy()` drops its index, so that pandas subtracts row by row instead of matching index labels. Last, it forecasts the last day, Monday 31 December 2012, from the last trend, `.iloc[-1]`, the value at the end of what is left after `dropna`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_decomposition():
    import pandas as pd

    # the registered riders of every day, with the date as the index
    _df = pd.read_csv("data/bike-sharing-day.csv.gz")
    _df["dteday"] = pd.to_datetime(_df["dteday"])
    _y = _df.set_index("dteday")["registered"]

    # 2. the trend: a centered moving average over one period, 7 days
    _trend = _y.rolling(7, center=True).mean()
    print(round(_trend.loc["2011-07-01"]), round(_trend.loc["2012-07-01"]))

    # 3. the season: the mean per weekday, shifted to add up to 0
    _rest = _y - _trend
    _season = _rest.groupby(_rest.index.dayofweek).mean()
    _season = _season - _season.mean()
    print(list(_season.round()))

    # 4. the remainder, and its lowest day
    _remainder = _rest - _season.loc[_rest.index.dayofweek].to_numpy()
    print(_remainder.idxmin(), round(_remainder.min()))

    # 5. a forecast for the last day: the last trend plus the season of a Monday
    print(round(_trend.dropna().iloc[-1] + _season.loc[0]), _y.loc["2012-12-31"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    On 1 July the trend was 3,594 bikes a day in 2011 and 4,958 in 2012: the scheme grew. The season runs from +411 on Thursday to −760 on Sunday, and the lowest remainder is Hurricane Sandy, 3,799 below trend and season. The forecast for Monday 31 December, the last trend, of Friday 28 December, plus the season of a Monday, is 1,476; registered riders rented 2,290. The last trend falls in the quiet week of Christmas, and a forecast this simple cannot know that the riders come back. How good a forecast is, and how to test it on the past, is the lesson *Testing a forecast on the past*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_moving_average():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A centered moving average by hand

        `moving_average(y, half)` should give the trend of the array `y`: for every day `i`, the mean of the `half` days before it, day `i` itself and the `half` days after it, NaN for the first and the last `half` days, where the window does not fit. The first lines make an array of NaN as long as `y` with `np.full`, and go over the days whose window fits. Replace the `...`. The check calls `moving_average(weeks, 3)`, eight weeks of registered riders from Monday 2 January 2012, and tries five more: a short made-up array with `half` 0, 1, 2 and 3, and five equal values. It compares with `y.rolling(2 * half + 1, center=True).mean()`.
        """),
        L.task("Let `moving_average` give the mean of the window around every day."),
    ])
    return


@app.cell(hide_code=True)
def exercise_moving_average():
    box_moving_average = L.code_box("moving-average", lines=7)
    box_moving_average
    return (box_moving_average,)


@app.cell(hide_code=True)
def check_moving_average(box_moving_average):
    result_moving_average = L.check("moving-average", box_moving_average.value)
    result_moving_average
    return (result_moving_average,)


@app.cell(hide_code=True)
def hints_moving_average():
    L.hints(
        "moving-average",
        [
            "The trend of day `i` is a mean over a stretch of `y`. Which days are in that stretch?",
            "The stretch runs from `half` days before `i` to `half` days after it, both included. A slice `y[a:b]` stops just before `b`.",
            "`trend[i] = np.mean(y[i - half:___])`.",
        ],
        why="With `half` 3, day 10 gets the mean of days 7 to 13, both included: `y[7:14]`, 7 values.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_season_of():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The season, position by position

        `season_of(rest, period)` should give the season: for every position `k` from 0 to `period - 1`, the mean of `rest[k]`, `rest[k + period]`, `rest[k + 2 * period]` and so on, NaN left out, with the `period` means shifted so that they add up to 0. Replace the two `...`. The check calls `season_of(rest, 7)`, with `rest` the registered riders minus their trend over 7 days, all 731 days, NaN for the first and the last 3. Position 0 is Saturday, because 1 January 2011 was one. It tries four more short arrays, two with NaN and one with `period` 1.
        """),
        L.task("Let `season_of` give the mean of every position in the period, shifted to add up to 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_season_of():
    box_season_of = L.code_box("season-of", lines=7)
    box_season_of
    return (box_season_of,)


@app.cell(hide_code=True)
def check_season_of(box_season_of):
    result_season_of = L.check("season-of", box_season_of.value)
    result_season_of
    return (result_season_of,)


@app.cell(hide_code=True)
def hints_season_of():
    L.hints(
        "season-of",
        [
            "`means[k]` is a mean over every `period`-th value of `rest`, starting at position `k`. Then every mean moves by the same amount, so that together they add up to 0.",
            "*The season* in *The idea* has a slice that takes every 7th value, and a mean that leaves NaN out. Seven numbers that add up to 0: what do you subtract from each?",
            "`means[k] = np.nanmean(rest[___::period])` and `return means - np.mean(___)`.",
        ],
        why="`rest[0::7]` is every Saturday, `rest[1::7]` every Sunday. The seven means are −560.6, −760.2, 14.1, 290.0, 332.2, 410.0 and 270.3, with a mean of −0.6; subtracting it gives −560.0, −759.6, 14.7, 290.6, 332.8, 410.6 and 270.9, which add up to 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_window():
    mo.md(r"""
    ### Exercise 3. How long a window
    """)
    return


@app.cell(hide_code=True)
def choice_window_cell():
    choice_window = L.choice(
        "window",
        question="The trend of the registered riders is a centered moving average over 7 days. Why 7, or a multiple of 7, such as 21 or 35?",
        options={
            "right": "Only then does every window hold every weekday equally often, so the week averages out of the trend",
            "long": "A shorter window is too bumpy; any window of 7 days or more works",
            "middle": "7 is odd, so every day has a middle and the window can be centered",
        },
        correct="right",
        why={
            "long": "Calm is not the point: a window of 9 days holds two weekdays twice, and lets part of the week back into the trend. Then the season spans 1,337 instead of 1,170. See the slider in *The idea*.",
            "middle": "An odd window does have a middle, but 5 is odd too: a window of 5 days misses two weekdays, the trend takes up part of the week, and the season spans 942 instead of 1,170. See the slider in *The idea*.",
            "right": "One whole period holds every position once, so the mean over it has no season left. 21 and 35 days hold whole weeks too, and give a calmer trend.",
        },
    )
    choice_window
    return (choice_window,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The trend is the rolling mean of [Time as an index](/course/open/ts-time-index?lang=en), moved to the middle of its window, and the season is a mean per group, as in `groupby`. The remainder is to a decomposition what a residual is to a model: what it leaves unexplained. The next lesson, *Does today look like yesterday? Autocorrelation*, finds the period of a series from the series itself; the forecast of trend plus season gets its test in *Testing a forecast on the past*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_window, result_moving_average, result_season_of):
    L.wrap_up(result_moving_average, result_season_of, choice_window)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
