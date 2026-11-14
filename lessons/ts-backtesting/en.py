# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ts-backtesting. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-backtesting", "en")
    C = L.checks   # the riders of every day, three forecasts backtested over 2012 for every horizon, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Does today look like yesterday? Autocorrelation](/course/open/ts-autocorrelation?lang=en) and one about [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lag():
    quick_lag = L.choice(
        "quick-lag",
        question="The bikes rented by registered riders every hour have an autocorrelation of 0.886 at lag 168. What does that say?",
        options={
            "right": "Every hour moves closely together with the same hour a week earlier",
            "share": "88.6% of the hours have as many riders as the same hour a week earlier",
            "rise": "The riders grow by 88.6% every week",
        },
        correct="right",
        why={
            "share": "r is not a share of equal hours: it says how well a rising straight line fits the pairs of an hour and the hour 168 earlier. See *Autocorrelation* in *The idea* in [Autocorrelation](/course/open/ts-autocorrelation?lang=en).",
            "rise": "A correlation says nothing about growth: a series that doubles every week and one that stays the same can both lie on a line with their past. See *The correlation coefficient* in *The idea* in [Autocorrelation](/course/open/ts-autocorrelation?lang=en).",
            "right": "A busy Monday 8:00 follows a busy Monday 8:00. Today that becomes a forecast: the same hour, or the same day, a week earlier.",
        },
    )
    quick_lag
    return (quick_lag,)


@app.cell(hide_code=True)
def quick_check_mae():
    quick_mae = L.choice(
        "quick-mae",
        question="A forecast for three days is 100 too low, 300 too high and 200 too low. What is its MAE?",
        options={
            "right": "200",
            "zero": "0",
            "largest": "300",
        },
        correct="right",
        why={
            "zero": "With their signs, the misses cancel: 100 − 300 + 200 = 0. The MAE takes them without their sign. See *The idea* in [How far off?](/course/open/ml-regression-metrics?lang=en)",
            "largest": "300 is the largest miss. The MAE is their mean size: (100 + 300 + 200) / 3. See *The idea* in [How far off?](/course/open/ml-regression-metrics?lang=en)",
            "right": "(100 + 300 + 200) / 3 = 200: on average 200 off, in the unit of what is forecast. Today it scores forecasts on days they have not seen.",
        },
    )
    quick_mae
    return (quick_mae,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A forecast is worth only as much as its misses on days it had not seen. Fit a model on all of 2012 and measure it on 2012, and it looks better than it will ever be, because it has seen the answers. The honest test makes every forecast the way it would have been made at the time: with the past only, and then checks it against what happened next. And the score means something only beside the score of a forecast that takes no effort at all.

    **An everyday example.** To judge a weather forecaster, you take what they said each evening about the next day, and compare it with the weather that came. A forecast written the day after does not count. And if "tomorrow will be like today" does as well, the forecaster adds nothing.
    """)
    return


@app.cell(hide_code=True)
def predict_yesterday_or_last_week():
    guess_day = L.guess(
        "day-ahead-guess",
        question="Forecast the registered riders of every day of 2012 one day ahead, in two ways: the day before, or the same weekday a week before. Which one misses less, measured by the MAE?",
        options={
            "right": "The day before",
            "week": "The same weekday a week before",
            "same": "They miss about equally",
        },
        reveal="**The day before: an MAE of 804, against 860.** A day is more like the day before than like the same day a week earlier: the autocorrelation at lag 1 is 0.832, at lag 7 0.764. But forecast a whole week ahead, and the day before has to stand in for all seven days, Saturday and Sunday included: its MAE grows to 1,340, while the week before keeps its 859.",
    )
    guess_day
    return (guess_day,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Only the past.** A forecast may use only what was known when it was made. The moment it is made is the **origin**; the days before it are the training days, the `h` days after it are the test days. `h` is how far ahead the forecast looks.

        **Backtesting.** Put the origin at Monday 2 January 2012, train on the days before it, forecast the week after it, and measure the MAE on that week. Then move the origin a week on and do it again: 52 origins, every Monday from 2 January to 24 December 2012. The mean of the 52 MAEs scores the forecast. Testing a forecast on the past like this is **backtesting**, and moving the origin on is a **rolling origin**. In Python, `range(first, len(y) - h + 1, h)` counts from `first` in steps of `h`, the third number, up to the last origin whose `h` test days still fit.

        **Two forecasts that take no effort.** The **naive forecast** says every day ahead is like the last day. The **seasonal naive forecast** says every day ahead is like the same day of the last period: with a period of 7 days, every Monday ahead is the last Monday. With `last` the last period of the training days, day `i` ahead is `last[i % period]`: `%` gives the remainder of `i` divided by `period`, so after the last day of the period it starts again at the first. A forecast that does not beat both is not worth its trouble. Next to them, this lesson tests the forecast of [Trend, season and the rest](/course/open/ts-trend-season?lang=en): the last trend plus the season.

        **MASE.** An MAE of 859 riders is good or bad depending on the series. The **MASE**, the mean absolute scaled error, divides the MAE by the MAE that the naive forecast makes one day ahead on the training days: here 510.6 riders. Below 1, a forecast misses less than "tomorrow is like today" did on the training days; and series of any size can be compared.

        **sMAPE.** The **sMAPE**, the symmetric mean absolute percentage error, takes every miss as a share of the true value and the forecast together: 2 × |true − forecast| / (|true| + |forecast|), and gives the mean of that in percent, from 0 to 200%.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **origin** | the moment a forecast is made: it may use only what came before | `origin` |
        | **backtesting** | testing a forecast on the past, as it would have been made at the time | |
        | **rolling origin** | moving the origin on, step by step, and forecasting again from each | `range(first, len(y) - h + 1, h)` |
        | **naive forecast** | every step ahead is the last value | `np.full(h, train[len(train) - 1])` |
        | **seasonal naive forecast** | every step ahead is the same step of the last period | `last[i % period]` |
        | `%` | the remainder of a division: `9 % 7` is 2 | `i % period` |
        | **MASE** | the MAE divided by the MAE of the naive forecast one step ahead on the training data | |
        | **sMAPE** | the mean of every miss as a share of the true value and the forecast together, in percent | |
        """),
        L.watch_out("A share breaks down near 0. On Monday 29 October 2012, Hurricane Sandy, registered riders rented 20 bikes, and the seasonal naive forecast said 6,228: that day alone counts 198.7% in the sMAPE, close to the most a day can count, 200%. On a day with almost nothing to count, every forecast is almost all wrong as a share. The MAE counts that day as 6,208 riders off, the same as any other miss of that size."),
        mo.md(r"""
        On the left, from Monday 4 June 2012: the riders of the four weeks before it, a vertical dashed line at the origin, the riders of the days after it, and the three forecasts for the next `h` days: the open gray dots are the naive forecast, the dashed line the seasonal naive one, the thick line trend plus season. On the right, the MAE of each over all origins of 2012 for that horizon. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_horizon_cell():
    def _frame(h):
        said = C.said(h, L.lang)
        alt = ("Left, a line chart of the riders of the four weeks before Monday 4 June 2012 and of the chosen number of days after "
               "it, with the naive, the seasonal naive and the trend plus season forecast for those days. Right, three bars: the MAE "
               "of each forecast over all origins of 2012. " + said)
        return (C.picture(h, {"left": "From Monday 4 June 2012", "right": "MAE over 2012", "ylabel": "registered riders",
                              "dates": ["7 May", "21 May", "4 Jun", "18 Jun", "2 Jul"],
                              "methods": ["naive", "seasonal", "trend+season"], "alt": alt}), said)

    explore_horizon = L.explore("horizon-days", "horizon (days)", _frame, start=7, values=C.HORIZONS, role="hyper",
                                task="Go from a horizon of 1 day to 28. Where does the naive forecast win, and how do the misses grow?")
    explore_horizon
    return


@app.cell(hide_code=True)
def text_after_horizon():
    mo.md(r"""
    A day ahead, the naive forecast beats the seasonal naive one, 804 against 860, as the guess showed. From 2 days on, the seasonal naive forecast wins, and its MAE hardly moves up to 14 days, from 859 to 868; the naive forecast misses more with every day: 1,340 a week ahead, 1,510 four weeks ahead. Trend plus season is the best at every horizon, from 669 a day ahead to 814 four weeks ahead. The further ahead, the more a forecast misses: say the horizon with every score.

    ### Recipe

    To test a forecast on the past:

    1. Choose the horizon you need and the first origin: here a week, from Monday 2 January 2012.
    2. At every origin, train only on what came before it, and forecast the `h` steps after it.
    3. Measure the error on those steps, and move the origin on by `h`.
    4. Take the mean over all origins, and do the same for the naive and the seasonal naive forecast.
    5. Give the MAE with the MASE, to compare across series; watch the sMAPE where values come close to 0.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell backtests the naive and the seasonal naive forecast a week ahead over 2012, with the steps of the recipe. `_backtest` takes a forecast as its argument: a function that gets the training days and gives the next 7. It gives back a list of two numbers, the mean MAE and the sMAPE. The cell prints the scale of the MASE, and then the MAE, the MASE and the sMAPE of both forecasts. Run the cell.
    """)
    return


@app.cell
def example_backtest():
    import numpy as np
    import pandas as pd

    _y = pd.read_csv("data/bike-sharing-day.csv.gz")["registered"].to_numpy()

    # 1. a week ahead, from Monday 2 January 2012, position 366
    _h = 7
    _first = 366


    def _naive(train):
        return np.full(_h, train[len(train) - 1])


    def _seasonal(train):
        _last = train[len(train) - 7:]
        _out = []
        for _i in range(_h):
            _out.append(_last[_i % 7])
        return np.array(_out)


    # 2. to 4. every origin: train on the days before it, test on the next 7, take the means
    def _backtest(forecast):
        _maes = []
        _smapes = []
        for _origin in range(_first, len(_y) - _h + 1, _h):
            _test = _y[_origin:_origin + _h]
            _guess = forecast(_y[:_origin])
            _maes.append(np.mean(abs(_test - _guess)))
            _smapes.append(np.mean(2 * abs(_test - _guess) / (abs(_test) + abs(_guess))))
        return [np.mean(_maes), 100 * np.mean(_smapes)]


    # 5. the scale of the MASE: the naive forecast a day ahead, on the training days
    _train = _y[:_first]
    _scale = np.mean(abs(_train[1:] - _train[:len(_train) - 1]))
    print(round(_scale, 1))
    for _forecast in [_naive, _seasonal]:
        _score = _backtest(_forecast)
        print(round(_score[0], 1), round(_score[0] / _scale, 2), round(_score[1], 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The naive forecast a day ahead missed 510.6 riders on average on the training days. A week ahead over 2012, the naive forecast has an MAE of 1,340.0, a MASE of 2.62 and an sMAPE of 35.4%; the seasonal naive forecast 858.5, 1.68 and 22.6%. Trend plus season, from the checks of this lesson, gets 739.1, 1.45 and 19.0%. None of the three gets below a MASE of 1: forecasting a week ahead is harder than tomorrow from today. The seasonal naive forecast beats the naive one in 42 of the 52 weeks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_seasonal_naive():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The seasonal naive forecast

        `seasonal_naive(train, h, period)` should forecast the `h` steps after `train`: step `i` ahead is the value at the same place in the last period. The first line takes that last period, `last`, and the loop goes over the steps ahead. Replace the `...`. The check calls `seasonal_naive(train, 10, 7)`, with `train` the registered riders up to Sunday 1 January 2012, so the forecast runs over a period and a bit, and tries four more: a horizon shorter than the period, a period of 2 and of 3, and a period of 1.
        """),
        L.task("Let `seasonal_naive` repeat the last period for every step ahead."),
    ])
    return


@app.cell(hide_code=True)
def exercise_seasonal_naive():
    box_seasonal_naive = L.code_box("seasonal-naive", lines=8)
    box_seasonal_naive
    return (box_seasonal_naive,)


@app.cell(hide_code=True)
def check_seasonal_naive(box_seasonal_naive):
    result_seasonal_naive = L.check("seasonal-naive", box_seasonal_naive.value)
    result_seasonal_naive
    return (result_seasonal_naive,)


@app.cell(hide_code=True)
def hints_seasonal_naive():
    L.hints(
        "seasonal-naive",
        [
            "`last` holds `period` values, one for every place in the period. Step `i` ahead takes one of them; after the last one, it starts again at the first.",
            "Which place in the period is step `i`? For `period` 7: steps 0 to 6 are places 0 to 6, step 7 is place 0 again, step 8 place 1. *Two forecasts that take no effort* in *The idea* has the sign that does this.",
            "`out.append(last[i ___ period])`.",
        ],
        why="With `period` 7 and `h` 10, the steps 0 to 9 take the places 0, 1, 2, 3, 4, 5, 6, 0, 1, 2 of `last`: 9 % 7 = 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_backtest():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A backtest with a rolling origin

        `backtest(y, first, h)` should give the MAE of the naive forecast for every origin, as a list: from `first`, in steps of `h`. At every origin, `train` must be what came before it, and `test` the `h` steps from the origin on. Replace the two `...`. The check calls `backtest(riders, 366, 7)`, the registered riders of all 731 days, from Monday 2 January 2012 a week at a time, and tries four more short arrays with other origins and horizons.
        """),
        L.task("Let `backtest` train on the past only, and test on the `h` steps after the origin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_backtest():
    box_backtest = L.code_box("backtest", lines=10)
    box_backtest
    return (box_backtest,)


@app.cell(hide_code=True)
def check_backtest(box_backtest):
    result_backtest = L.check("backtest", box_backtest.value)
    result_backtest
    return (result_backtest,)


@app.cell(hide_code=True)
def hints_backtest():
    L.hints(
        "backtest",
        [
            "At every origin, the forecast may know what came before the origin, and nothing from the origin on. The test days are the ones it forecasts.",
            "`train` is a slice of `y` that stops just before `origin`; `test` starts at `origin` and holds `h` values. A slice `y[a:b]` stops just before `b`.",
            "`train = y[:___]` and `test = y[origin:___]`.",
        ],
        why="With `y` = [1, 2, ..., 10], `first` 4 and `h` 3, the origins are 4 and 7. At 4, `train` is [1, 2, 3, 4] and `test` [5, 6, 7]; the forecast is 4, 4, 4, and the MAE (1 + 2 + 3) / 3 = 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mase():
    mo.md(r"""
    ### Exercise 3. What the MASE adds
    """)
    return


@app.cell(hide_code=True)
def choice_mase_cell():
    choice_mase = L.choice(
        "mase",
        question="Over 2012, the seasonal naive forecast a week ahead has an MAE of 859 riders and a MASE of 1.68. What does the 1.68 add to the 859?",
        options={
            "right": "It compares the 859 with the naive forecast a day ahead on the training days, 510.6: a week ahead, this forecast misses 1.68 times as much",
            "percent": "The forecast misses by 1.68% of the riders",
            "large": "It counts large misses more, as the RMSE does",
        },
        correct="right",
        why={
            "percent": "A share of the riders is what the sMAPE gives, here 22.6%. The MASE divides by an MAE, so it has no unit at all. See *MASE* in *The idea*.",
            "large": "The RMSE squares the misses; the MASE divides the MAE, which counts every miss by its size, by another MAE. See *MASE* in *The idea*.",
            "right": "1.68 says how this forecast does against a fixed yardstick, without riders as its unit, so a station that rents 50 bikes a day and a city that rents 5,000 can be compared.",
        },
    )
    choice_mase
    return (choice_mase,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The naive and the seasonal naive forecast are the baselines of [Always beat a simple baseline](/course/open/ml-baselines?lang=en) for time series, scored with the MAE of [How far off?](/course/open/ml-regression-metrics?lang=en). Which of the two wins was in the autocorrelation already: lag 1 is the strongest a day ahead, lag 7 for a week. The next lesson, *Forecasting demand with trees*, turns the past into columns, such as the riders one and two weeks earlier, lets gradient boosting learn from them, and tests it with this backtest against the seasonal naive forecast.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_mase, result_backtest, result_seasonal_naive):
    L.wrap_up(result_seasonal_naive, result_backtest, choice_mase)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
