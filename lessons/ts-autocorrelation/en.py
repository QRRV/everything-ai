# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ts-autocorrelation. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-autocorrelation", "en")
    C = L.checks   # the riders of every hour on a grid of hours, their autocorrelation at every lag, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Trend, season and the rest](/course/open/ts-trend-season?lang=en) and one about [Two columns together: an association is not a cause](/course/open/data-two-columns?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_season():
    quick_season = L.choice(
        "quick-season",
        question="`rest` is a series of days minus its trend, and its first day is a Monday. What is `np.nanmean(rest[2::7])`?",
        options={
            "right": "The mean of all Wednesdays: the season of Wednesday, before the seven means are shifted to add up to 0",
            "stretch": "The mean of the days at positions 2 to 7",
            "tuesday": "The mean of all Tuesdays",
        },
        correct="right",
        why={
            "stretch": "`rest[2:7]` would be one stretch. `::7` steps: positions 2, 9, 16 and so on, every 7th day from position 2. See *The season* in *The idea* in [Trend, season and the rest](/course/open/ts-trend-season?lang=en).",
            "tuesday": "Positions count from 0: position 0 is the Monday, 1 the Tuesday, 2 the Wednesday. See *The season* in *The idea* in [Trend, season and the rest](/course/open/ts-trend-season?lang=en).",
            "right": "Every 7th day from position 2: all Wednesdays. Today a series is compared with itself a number of steps earlier, and the same week shows up again.",
        },
    )
    quick_season
    return (quick_season,)


@app.cell(hide_code=True)
def quick_check_association():
    quick_association = L.choice(
        "quick-association",
        question="On warm days the ice cream stand sells more, and more bikes are rented. Does renting a bike make people buy ice cream?",
        options={
            "right": "We cannot tell from this: warm weather drives both, so they move together either way",
            "yes": "Yes: the two columns move together, so one causes the other",
            "never": "No: two columns that move together never have anything to do with each other",
        },
        correct="right",
        why={
            "yes": "Moving together is not causing: a third column, here the weather, can drive both. See *The idea* in [Two columns together](/course/open/data-two-columns?lang=en).",
            "never": "Columns that move together often do have something to do with each other, sometimes through a third column. Moving together alone just does not say what causes what. See *The idea* in [Two columns together](/course/open/data-two-columns?lang=en).",
            "right": "The weather is a confounder. Today, one number says how strongly two columns move together, and then a series is set beside itself a few steps earlier.",
        },
    )
    quick_association
    return (quick_association,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To forecast how many bikes registered riders will rent in the next hour, the obvious place to look is the past. But which part of it? The hour just before, the same hour yesterday, or the same hour last week? The answer is in how much a series looks like itself some steps earlier. It also shows the season of a series without being told the period, and it warns when a trend makes everything look alike.

    **An everyday example.** Will it be warm tomorrow? Today's temperature is a good guess: weather changes slowly, so a day looks like the day before. Whether it rains tomorrow is much harder to guess from today. Some series remember their past well, others hardly at all.
    """)
    return


@app.cell(hide_code=True)
def predict_earlier_hour():
    guess_hour = L.guess(
        "earlier-hour-guess",
        question="For every hour of 2011 and 2012, set the bikes rented by registered riders beside the bikes rented some hours earlier. Which earlier hour moves most closely together with the hour now?",
        options={
            "hour": "The hour just before",
            "day": "The same hour a day earlier",
            "right": "The same hour a week earlier",
        },
        reveal="**The same hour a week earlier.** With the number of this lesson, r, it is 0.886 for a week earlier, 0.815 for a day earlier and 0.802 for an hour earlier. The hour before is close in time, but on a workday morning the riders climb fast: on average 98 at 6:00, 278 at 7:00 and 455 at 8:00. Monday 8:00 last week was a Monday morning too.",
    )
    guess_hour
    return (guess_hour,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The correlation coefficient.** For pairs of numbers, such as the temperature and the riders of every day, the **correlation coefficient** r says how well a straight line fits them, rising or falling:

        $$r = \frac{\operatorname{mean}\big((x - \operatorname{mean}(x)) \cdot (y - \operatorname{mean}(y))\big)}{\operatorname{std}(x) \cdot \operatorname{std}(y)}$$

        Every pair adds the product of its distances to the two means. That product is positive when both values are above their mean or both below, and negative when one is above and the other below. Dividing by the two standard deviations makes r free of units: it lies between −1 and 1. r = 1 means every pair lies on one rising line, −1 on one falling line, and 0 that no straight line fits, which can still be a curve: `x` and `x ** 2` from −3 to 3 give 0. Temperature and registered riders give 0.540. In NumPy, `np.corrcoef(x, y)` gives a table of 2 by 2 correlations, of `x` and `y` with themselves and with each other, so `np.corrcoef(x, y)[0, 1]` is r.

        **Lag.** A **lag** of `k` steps is the value `k` steps earlier, what `y.shift(k)` puts beside every value.

        **Autocorrelation.** The **autocorrelation** at lag `k` is the correlation of a series with itself `k` steps earlier: every value from position `k` on, paired with the value `k` steps before it. In pandas: `y.autocorr(k)`. Draw it for every lag as a bar, and the season shows itself: for the riders of every hour, the bars peak at 24 hours, 0.815, and higher still at 168 hours, a week, 0.886, while 12 hours, from day to night, gives −0.021.

        **A trend makes every lag alike.** The registered riders of every day have an autocorrelation between 0.57 and 0.83 at every lag from 1 to 28 days: a day in a busy month lies close to all the days around it. The rest of the last lesson, the riders minus their trend, shows the week: 0.390 at 7 days and −0.501 at 3.

        **Stationarity.** A series is **stationary** when its mean and its spread stay the same over time: any stretch of it looks like any other. The registered riders are not: they rented 2,728 bikes a day on average in 2011, and 4,581 in 2012. Most forecasting methods assume a stationary series, which is why the trend and the season are taken off first.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **correlation coefficient** | how well a straight line fits pairs of numbers, from −1 (falling) through 0 (no line) to 1 (rising) | `np.corrcoef(x, y)[0, 1]` |
        | r | the correlation coefficient | |
        | **lag** | a number of steps back in time | `y.shift(k)` |
        | **autocorrelation** | the correlation of a series with itself a lag earlier | `y.autocorr(k)` |
        | **stationary** | a series whose mean and spread stay the same over time | |
        | `np.corrcoef` | the correlations of two arrays, with themselves and with each other, as a table of 2 by 2 | `np.corrcoef(x, y)` |
        """),
        L.watch_out("A lag of `k` rows is a lag of `k` steps only on a grid. The hours of the bike data have 165 gaps; on the table itself, a row 168 rows back is a week back only if no hour in between is missing. On the table, `autocorr(168)` gives 0.823; on `asfreq(\"h\")`, where every hour has its place, 0.886. Put a series on its grid before you compare it with its past."),
        mo.md(r"""
        On the left, every dot is one hour: the riders `k` hours earlier across, and the riders at that hour up. On the right, the autocorrelation at every lag up to two weeks, the lag you choose drawn thick. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lag_cell():
    def _frame(lag):
        said = C.said(lag, L.lang)
        alt = ("Left, a scatter plot of the bikes rented by registered riders in an hour against the bikes rented the chosen "
               "number of hours earlier, both between 0 and 900. Right, the autocorrelation as a thin bar for every lag from 1 to "
               "336 hours, with peaks at every 24 hours and the highest at 168, the chosen lag drawn thick. " + said)
        return (C.picture(lag, {"left": "Riders now and k hours earlier", "right": "Autocorrelation", "xlabel": "k hours earlier",
                                "ylabel": "now", "lag": "lag (hours)", "alt": alt}), said)

    explore_lag = L.explore("lag-hours", "lag (hours)", _frame, start=24, values=C.LAGS, role="hyper",
                            task="Go from a lag of 1 hour to 336. At which lags do the dots lie close to a line, and at which are they a cloud?")
    explore_lag
    return


@app.cell(hide_code=True)
def text_after_lag():
    mo.md(r"""
    At 1 hour the dots lie close to a line, at 3 hours they spread out, and at 6 and 12 hours they are a cloud: r is about 0. Then the day comes back: at 24 hours, r is 0.815 again, at 167 hours 0.717, and at 168, a whole week, 0.886, the highest of all lags up to two weeks. 336 hours, two weeks, is almost as high: 0.884. The peaks of the bars at every 24 hours and every 168 are the two seasons of this series, the day and the week.

    ### Recipe

    To see how a series hangs together with its past:

    1. Put the series on a grid of equal steps, such as `asfreq("h")`, so that a lag of `k` rows is `k` steps.
    2. Compute the autocorrelation for every lag up to a few periods: `y.autocorr(k)`.
    3. Peaks at a lag and its multiples show a season with that period.
    4. An autocorrelation that stays high at every lag means a trend: take the trend off and look again.
    5. Compare the mean and the spread of stretches, such as years, to see whether the series is stationary.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell puts the riders of every hour on a grid of hours: `pd.to_timedelta(_h["hr"], unit="h")` turns the column of hour numbers into lengths of time, added to the dates. It prints the autocorrelation at 1, 24 and 168 hours. Then it takes the riders of every day and their rest after a trend over 7 days, and prints the autocorrelation of both at 3 and 7 days. Last, it compares 2011 with 2012: the first 365 days and the rest, their means and the standard deviations of the rest. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_autocorrelation():
    import pandas as pd

    # 1. the riders of every hour, on a grid of hours
    _h = pd.read_csv("data/bike-sharing-hour.csv.gz")
    _when = pd.to_datetime(_h["dteday"]) + pd.to_timedelta(_h["hr"], unit="h")
    _s = _h.set_index(_when)["registered"].asfreq("h")

    # 2. and 3. the autocorrelation an hour, a day and a week back
    print(round(_s.autocorr(1), 3), round(_s.autocorr(24), 3), round(_s.autocorr(168), 3))

    # 4. the riders of every day, with and without their trend
    _y = pd.read_csv("data/bike-sharing-day.csv.gz")["registered"]
    _rest = _y - _y.rolling(7, center=True).mean()
    print(round(_y.autocorr(3), 3), round(_y.autocorr(7), 3))
    print(round(_rest.autocorr(3), 3), round(_rest.autocorr(7), 3))

    # 5. stationary? the mean of the riders and the spread of the rest, 2011 against 2012
    print(round(_y[:365].mean()), round(_y[365:].mean()))
    print(round(_rest[:365].std()), round(_rest[365:].std()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    An hour, a day and a week back give 0.802, 0.815 and 0.886. The riders of every day give 0.646 at 3 days and 0.764 at 7: the trend keeps both high. Without the trend, 3 days gives −0.501 and 7 days 0.390: a Thursday is unlike the Sunday 3 days later, and like the Thursday a week later. The mean grew from 2,728 a day in 2011 to 4,581 in 2012. Taking the trend off fixes the mean, but the spread of the rest grew too, from 529 to 901: the rest is not stationary either.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_correlation():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The correlation coefficient by hand

        `correlation(x, y)` should give the correlation coefficient of two arrays of the same length. The first lines give the distances of every value to its mean, `dx` and `dy`. Replace the `...` with r. The check calls `correlation(temp, riders)`, the temperature in °C and the registered riders of all 731 days, and tries four more: a rising line, a falling line, the bowl of `x` and `x ** 2`, and five made-up pairs. It compares with `np.corrcoef`.
        """),
        L.task("Let `correlation` give r: how well a straight line fits the pairs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_correlation():
    box_correlation = L.code_box("correlation", lines=6)
    box_correlation
    return (box_correlation,)


@app.cell(hide_code=True)
def check_correlation(box_correlation):
    result_correlation = L.check("correlation", box_correlation.value)
    result_correlation
    return (result_correlation,)


@app.cell(hide_code=True)
def hints_correlation():
    L.hints(
        "correlation",
        [
            "Above the line of the formula: one mean, of products. Below it: two standard deviations.",
            "The products are `dx * dy`, one per pair. `np.std` gives a standard deviation.",
            "`return np.mean(___) / (np.std(x) * ___)`.",
        ],
        why="For the temperature and the riders, the mean of the products is 6,314.8, and the standard deviations are 7.5 °C and 1,559.2 riders: 6,314.8 / (7.5 × 1,559.2) = 0.540.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_autocorrelation():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The autocorrelation at lag k

        `autocorrelation(y, k)` should give the autocorrelation of the array `y` at lag `k`: `now` is every value from position `k` on, `before` is the value `k` steps before each of those, and the last line gives the correlation of the two. Replace the two `...` with slices. The check calls `autocorrelation(riders, 7)`, the registered riders of all 731 days, and tries five more: lag 1 of the riders, a rising series, a series that goes up and down every step at lag 1 and 2, and lag 0. It compares with what pandas gives, `pd.Series(y).autocorr(k)`.
        """),
        L.task("Let `autocorrelation` pair every value with the value `k` steps earlier."),
    ])
    return


@app.cell(hide_code=True)
def exercise_autocorrelation():
    box_autocorrelation = L.code_box("autocorrelation", lines=6)
    box_autocorrelation
    return (box_autocorrelation,)


@app.cell(hide_code=True)
def check_autocorrelation(box_autocorrelation):
    result_autocorrelation = L.check("autocorrelation", box_autocorrelation.value)
    result_autocorrelation
    return (result_autocorrelation,)


@app.cell(hide_code=True)
def hints_autocorrelation():
    L.hints(
        "autocorrelation",
        [
            "`now` and `before` must be equally long, and `before[i]` must be the value `k` steps before `now[i]`. The first `k` values have nothing before them; the last `k` values are never `k` steps before another.",
            "`now` leaves out the first `k` values, `before` the last `k`. A slice `y[a:b]` can leave out either end; `len(y)` is where `y` ends.",
            "`now = y[___:]` and `before = y[:len(y) - ___]`.",
        ],
        why="With `y` = [1, 2, 3, 4, 5] and `k` = 2: `now` is [3, 4, 5] and `before` is [1, 2, 3], so 3 is paired with 1, 4 with 2 and 5 with 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stationary():
    mo.md(r"""
    ### Exercise 3. Stationary or not
    """)
    return


@app.cell(hide_code=True)
def choice_stationary_cell():
    choice_stationary = L.choice(
        "stationary",
        question="The rest of the registered riders, after the trend over 7 days, has a mean close to 0 in every quarter of 2011 and 2012. Its standard deviation grows from 342 in the first quarter of 2011 to 1,035 in the last of 2012. Is the rest stationary?",
        options={
            "right": "No: its spread changes over time, and a stationary series keeps both its mean and its spread",
            "mean": "Yes: its mean stays close to 0",
            "trend": "Yes: the trend is gone, and that is what stationary means",
        },
        correct="right",
        why={
            "mean": "A steady mean is half of it. A stationary series keeps its spread too, and this spread triples, from 342 to 1,035: at the end of 2012, the days lie much further from their trend than at the start of 2011. See *Stationarity* in *The idea*.",
            "trend": "Taking off the trend makes the mean steady, but the spread of the rest still grows over the two years. Stationary asks for both. See *Stationarity* in *The idea*.",
            "right": "The spread of the rest grew over the two years, from 342 in the first quarter to 1,035 in the last. A forecast that assumes the same spread everywhere is too sure where the spread is large and too careful where it is small.",
        },
    )
    choice_stationary
    return (choice_stationary,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The correlation coefficient puts a number on the scatter plots of [Two columns together](/course/open/data-two-columns?lang=en), and it says no more about cause than they did. The autocorrelation finds the period that [Trend, season and the rest](/course/open/ts-trend-season?lang=en) had to be told, and the rest of that lesson is what it reads without a trend. The next lesson, *Testing a forecast on the past*, turns the two strongest lags into the two simplest forecasts: the value a step earlier, and the value a season earlier.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stationary, result_autocorrelation, result_correlation):
    L.wrap_up(result_correlation, result_autocorrelation, choice_stationary)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
