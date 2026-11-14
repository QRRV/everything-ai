# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ts-forecast-demand. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-forecast-demand", "en")
    C = L.checks   # the riders of every hour, the tree backtested a week ahead for every step of the slider, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Testing a forecast on the past](/course/open/ts-backtesting?lang=en) and one about [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_origin():
    quick_origin = L.choice(
        "quick-origin",
        question="A backtest puts its origin at Monday 4 June 2012 and forecasts the week after it. Which days may the forecast train on?",
        options={
            "right": "Every day before 4 June",
            "around": "Every day of 2011 and 2012 except that week",
            "week": "Only the week before 4 June",
        },
        correct="right",
        why={
            "around": "The days after that week are the future too: a forecast made on 4 June cannot know them. See *Only the past* in *The idea* in [Testing a forecast on the past](/course/open/ts-backtesting?lang=en).",
            "week": "The seasonal naive forecast uses only the last week, but a forecast may train on everything that came before the origin. See *Only the past* in *The idea* in [Testing a forecast on the past](/course/open/ts-backtesting?lang=en).",
            "right": "Only what was known at the origin. Today the same holds for every column of a table: it may only hold what was known at the origin.",
        },
    )
    quick_origin
    return (quick_origin,)


@app.cell(hide_code=True)
def quick_check_missing():
    quick_missing = L.choice(
        "quick-missing",
        question="Some rows of a table have NaN in a feature. What does `HistGradientBoostingRegressor` do with them?",
        options={
            "right": "It handles the missing values by itself",
            "error": "It stops with an error, so drop those rows first",
            "zero": "It counts them as 0",
        },
        correct="right",
        why={
            "error": "Not this model: [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en) says that it handles missing values by itself.",
            "zero": "A 0 would be a value like any other, and a wrong one. [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en) says that this model handles missing values by itself.",
            "right": "Today that matters: the first two weeks have no value two weeks earlier, and 165 hours have no row at all. A missing label is another matter, as you will see.",
        },
    )
    quick_missing
    return (quick_missing,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    How many bikes, meals, megawatts or nurses tomorrow and next week: forecasts of demand plan stock, staff and power. Demand depends on more than its own past: on the hour, the day of the week, a holiday, the weather. The forecasts of the previous lesson only repeat the past. A model such as gradient boosting can learn from all of it at once, once the past is turned into the columns of a table. But a table also makes it easy to cheat without noticing: one column with a value from after the origin, and the backtest looks far better than the forecast will ever be.

    **An everyday example.** A bakery plans Saturday's bread on Wednesday. It knows what it sold last Saturday and the one before, that Saturday is market day, and that Monday is a holiday. What it will sell on Thursday and Friday, however much that would help, it does not know yet.
    """)
    return


@app.cell(hide_code=True)
def predict_tree_cell():
    guess_tree = L.guess(
        "tree-guess",
        question="Forecast a week ahead, an hour at a time, from every Monday from 2 January to 24 December 2012. The seasonal naive forecast, the same hour a week earlier, misses by 46.8 riders an hour on average. A tree learns from the riders one and two weeks earlier, the hour, the weekday, the month and whether a day is a working day. How much does it miss?",
        options={
            "worse": "More than 46.8",
            "right": "About 42: a tenth less",
            "half": "About 23: half",
        },
        reveal="**42.0, a tenth less.** But the plain mean of the same hour one and two weeks earlier, without any model, gets 42.1. Over the whole year, the tree adds almost nothing to the two weeks. It earns its place on the days a simple rule does not see coming, such as holidays, and with columns a simple rule cannot use, such as the weather.",
    )
    guess_tree
    return (guess_tree,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **From a series to a table.** A tree learns from a table: rows with features and a label. For a forecast, every hour becomes a row, its riders are the label, and the features are what is known about that hour beforehand.

        **Lag features.** The riders of the same series `k` hours earlier make a **lag feature**, `lag_k`. On the grid of hours of `asfreq("h")`, `y.shift(k)` moves every value `k` rows later, so each row gets the hour `k` before it, as in [Time as an index](/course/open/ts-time-index?lang=en). The autocorrelation of [Does today look like yesterday?](/course/open/ts-autocorrelation?lang=en) says which lags are worth a column: 0.886 at lag 168, a week.

        **Only lags the horizon allows.** A forecast from Monday 0:00 for the week ahead, `h` = 168 hours, must fill in `lag_k` for every hour of that week with what is known at the origin. The last hour, Sunday 23:00, is 167 hours after the origin; `k` hours before it is before the origin only when `k` is at least 168. So a lag must be at least the horizon: `k >= h`. A shorter lag, such as `lag_24`, is known for Monday, but from Tuesday on it holds hours of the week itself, the ones the forecast is for.

        **Calendar features.** What a date says about itself is known as far ahead as you like: the hour, `index.hour`, from 0 to 23; the weekday, `index.dayofweek`, 0 for Monday as in [Trend, season and the rest](/course/open/ts-trend-season?lang=en); the month, `index.month`, from 1 to 12; and whether a day is a working day, which the table of the course has as `workingday`: 0 on weekends and holidays. Holidays are known years ahead. These are **calendar features**.

        **A tree, tested as before.** `HistGradientBoostingRegressor` of [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en) learns from the table: at every origin, from the rows before it whose riders are known. A row without riders has no label to learn from, so it goes: `dropna(subset=["registered"])` drops only the rows with NaN in that column. A missing lag the model handles by itself. The backtest is the one of [Testing a forecast on the past](/course/open/ts-backtesting?lang=en): 52 origins, every Monday from 2 January to 24 December 2012, a week ahead each, scored with the mean of their MAEs.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **lag feature** | a column with the value of the series `k` steps earlier | `y.shift(k)` |
        | **calendar feature** | a column with what a date says about itself, known ahead | `y.index.dayofweek` |
        | `hour`, `month` | the hour, 0 to 23, and the month, 1 to 12, of every moment in a datetime index | `y.index.hour`, `y.index.month` |
        | `dropna(subset=[...])` | the table without the rows that have NaN in the columns named; NaN elsewhere stays | `t.dropna(subset=["registered"])` |
        | `threadpool_limits(1)` | inside its `with` block, scikit-learn calculates on one core | `with threadpool_limits(1):` |
        """),
        L.watch_out("A table can look ahead without any error. Put `lag_1` and `lag_24` in the table of a forecast a week ahead, and the backtest says 24.4 riders an hour, about half of the seasonal naive forecast's 46.8. That number is false: from the second hour of the week on, `lag_1` is an hour of the week itself, and from Tuesday on, `lag_24` is too. A forecast made on Monday 0:00 has neither. Ask of every column: was it known at the origin, for every hour of the horizon?"),
        mo.md(r"""
        The slider adds columns, step by step. Step 1: only the calendar. Step 2: only `lag_168`. Step 3: `lag_168` and `lag_336`. Step 4: both lags and the calendar. Step 5: also the weather as it turned out, temperature, kind of weather and humidity, which a forecast made a week earlier cannot know: it shows what a perfect weather forecast would be worth. On the left, the week from Monday 4 June 2012: the riders as a gray line, the seasonal naive forecast dashed, the tree of the step thick. On the right, the MAE of every step over the 52 weeks of 2012, which the course measured beforehand with the backtest of the worked example, and that of the seasonal naive forecast as a dashed line. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    def _frame(step):
        said = C.said(step, L.lang)
        alt = ("Left, a line chart of the riders of every hour of the week from Monday 4 June 2012, with the seasonal naive "
               "forecast and the forecast of the tree on the columns of this step. Right, five bars: the MAE of every step "
               "over the 52 weeks of 2012, with the seasonal naive forecast as a dashed line. " + said)
        return (C.picture(step, {"left": "Week of 4 June 2012", "right": "MAE over 2012", "ylabel": "registered riders",
                                 "step": "step", "seasonal": "seasonal naive", "lang": "en", "alt": alt}), said)

    explore_step = L.explore("feature-step", "step", _frame, start=1, values=[1, 2, 3, 4, 5], role="hyper",
                             task="Go from step 1 to step 5. Which columns lower the MAE, and which hardly do?")
    explore_step
    return


@app.cell(hide_code=True)
def text_after_steps():
    mo.md(r"""
    With only the calendar, the tree misses by 70.2 riders an hour, and too low: on average 58.7 riders below the truth. It knows when people ride, not how many ride now: at least half of its training hours are from 2011, when an hour had 115 riders on average, against 192 in 2012, and none of its columns says which year it is. `lag_168` brings the level, but alone the tree does worse than the seasonal naive forecast, which is that very column: 49.5 against 46.8. Two weeks do better, 42.1. The calendar adds almost nothing over the whole year, 42.0, but on the 11 holidays of 2012 the tree misses by 69.8 riders an hour, against 97.5 with the lags alone. That is `workingday`, which says the day is a holiday, something the lags cannot know: with the hour, the weekday and the month but without `workingday`, the tree still misses by 93.9 on the holidays. The weather as it turned out takes the MAE to 34.2: about what a perfect weather forecast adds here.

    ### Recipe

    To forecast with a tree:

    1. Put the series on a grid of time with `asfreq`, so that `k` rows are `k` steps.
    2. Choose the horizon `h`, and make lag features with lags of at least `h`.
    3. Add calendar features from the index, and whatever else is known at the origin, such as holidays.
    4. At every origin, train on the rows before it whose label is known, and forecast the `h` rows after it.
    5. Compare with the seasonal naive forecast and with a plain mean of the lags: the tree has to beat both.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell builds the table of the recipe, with `lag_168` and `lag_336`, the hour, the weekday, the month and `workingday`, and backtests the tree a week ahead from every Monday from 2 January to 24 December 2012. Row 8784 is Monday 2 January 2012, 0:00: the 366 days before it times 24 hours. Beside the tree, it scores the seasonal naive forecast, which is the column `lag_168` itself, and the mean of the two lags, which needs no model. `threadpool_limits(1)`, from threadpoolctl, which comes with scikit-learn, lets scikit-learn calculate on one core inside its `with` block. On a busy computer, cores that wait for each other cost more than they win: in a test of the course, with another program keeping one core busy, the 52 models took 67 seconds on four cores and 5 on one, with the same forecasts. The cell prints the number of hours in the table and how many of them have no riders, then the MAE of the seasonal naive forecast, of the mean of two weeks and of the tree. Run the cell.
    """)
    return


@app.cell
def example_forecast():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    _rows = pd.read_csv("data/bike-sharing-hour.csv.gz")
    _when = pd.to_datetime(_rows["dteday"]) + pd.to_timedelta(_rows["hr"], unit="h")
    _hours = _rows.set_index(_when).asfreq("h")   # 1. every hour a row, with NaN where the table had none

    # 2. and 3. the label, two lags of at least a week, and the calendar
    _table = pd.DataFrame(index=_hours.index)
    _table["registered"] = _hours["registered"]
    _table["lag_168"] = _hours["registered"].shift(168)
    _table["lag_336"] = _hours["registered"].shift(336)
    _table["hour"] = _hours.index.hour
    _table["weekday"] = _hours.index.dayofweek
    _table["month"] = _hours.index.month
    _table["workingday"] = _hours["workingday"]
    _columns = ["lag_168", "lag_336", "hour", "weekday", "month", "workingday"]
    print(len(_table), _table["registered"].isna().sum())

    # 4. every Monday of 2012: train on the hours before it that have riders, forecast the 168 after it
    _h = 168
    _tree = []
    _seasonal = []
    _two = []
    with threadpool_limits(1):   # scikit-learn on one core: see the text above the cell
        for _origin in range(8784, len(_table) - _h + 1, _h):
            _train = _table.iloc[:_origin].dropna(subset=["registered"])
            _test = _table.iloc[_origin:_origin + _h]
            _model = HistGradientBoostingRegressor(random_state=0)
            _model.fit(_train[_columns], _train["registered"])
            _y = _test["registered"]
            _tree.append(np.nanmean(abs(_y - _model.predict(_test[_columns]))))
            # 5. the seasonal naive forecast is the column lag_168; the mean of the two lags needs no model
            _seasonal.append(np.nanmean(abs(_y - _test["lag_168"])))
            _two.append(np.nanmean(abs(_y - (_test["lag_168"] + _test["lag_336"]) / 2)))
    print(round(np.mean(_seasonal), 1), round(np.mean(_two), 1), round(np.mean(_tree), 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The table has 17,544 hours, 165 of them without riders. A week ahead over 2012, the seasonal naive forecast misses by 46.8 riders an hour, the mean of two weeks by 42.1, and the tree by 42.0; the tree beats the seasonal naive forecast in 34 of the 52 weeks. Over the whole year, the two lags do the work. The tree's own gain is where it combines columns that a rule of thumb cannot, such as the holidays of `workingday`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lag_features():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Lag features

        `lag_features(y, lags, h)` should give a table with the rows of `y` and a column `lag_k` for every lag `k` in `lags` that a forecast `h` steps ahead may use, and no other. The first lines make the empty table and go over the lags; replace the `...` with the body of the loop. The check calls `lag_features(riders, [1, 24, 168, 336], 168)`, with `riders` the registered riders of every hour on the grid of hours, and tries four short Series with other lags and horizons.
        """),
        L.task("Let `lag_features` make a column for every lag the horizon allows, and no other."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lag_features():
    box_lag_features = L.code_box("lag-features", lines=9)
    box_lag_features
    return (box_lag_features,)


@app.cell(hide_code=True)
def check_lag_features(box_lag_features):
    result_lag_features = L.check("lag-features", box_lag_features.value)
    result_lag_features
    return (result_lag_features,)


@app.cell(hide_code=True)
def hints_lag_features():
    L.hints(
        "lag-features",
        [
            "Two things for every lag: does the horizon allow it, and if so, a new column. *Only lags the horizon allows* in *The idea* says which lags may stay.",
            "A lag `k` may stay when it is at least `h`. Its column is `y` with every value moved `k` rows later, under the name `lag_` followed by the number `k`; an f-string makes that name.",
            "`if k >= ___:` and, inside it, `table[f\"lag_{k}\"] = y.shift(___)`.",
        ],
        why="A week ahead, `h` is 168: `lag_1` and `lag_24` go, `lag_168` and `lag_336` stay. For Sunday 23:00, 167 hours after the origin, `lag_168` is the Sunday 23:00 a week earlier: the last hour before the origin.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_tree_forecast():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A forecast by a tree

        `tree_forecast(table, columns, origin, h)` should train a tree on the rows before `origin` whose `registered` is known, with `columns` as features and `registered` as label, and forecast the `h` rows from `origin` on. The lines for `test`, the model and `predict` are there; replace the three `...`. The check calls `tree_forecast(table, columns, 12480, 168)`: the table of the worked example, its six columns, and the origin Monday 4 June 2012, 0:00. It also tries three cases on two made-up tables, one of them with gaps.
        """),
        L.task("Let `tree_forecast` learn only from the rows before the origin that have riders."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tree_forecast():
    box_tree_forecast = L.code_box("tree-forecast", lines=9)
    box_tree_forecast
    return (box_tree_forecast,)


@app.cell(hide_code=True)
def check_tree_forecast(box_tree_forecast):
    result_tree_forecast = L.check("tree-forecast", box_tree_forecast.value)
    result_tree_forecast
    return (result_tree_forecast,)


@app.cell(hide_code=True)
def hints_tree_forecast():
    L.hints(
        "tree-forecast",
        [
            "The model may learn from what was known at the origin: the rows before it. And it can only learn from a row whose label it has.",
            "`train` is the rows of `table` up to `origin`, as in the backtest of the previous lesson, without the rows where `registered` is NaN; `dropna` with `subset` drops only those. `fit` takes the features first, then the label.",
            "`train = table.iloc[:___].dropna(subset=[___])` and `model.fit(train[___], train[___])`.",
        ],
        why="At origin 12,480, `train` holds the 12,355 hours before Monday 4 June 2012 that have riders: 125 of the 12,480 have none. The model learns from their six columns and forecasts the 168 hours of the week from the same six columns of `test`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_known():
    mo.md(r"""
    ### Exercise 3. What a forecast may know
    """)
    return


@app.cell(hide_code=True)
def choice_known_cell():
    choice_known = L.choice(
        "known-columns",
        question="From Monday 0:00, a tree forecasts every hour of the week ahead. Which of these columns may its table hold?",
        options={
            "right": "`workingday`: whether each day is a working day",
            "lag24": "`lag_24`: the riders 24 hours earlier",
            "temp": "`temp`: the temperature of every hour, as it turned out",
        },
        correct="right",
        why={
            "lag24": "For Monday, `lag_24` is the Sunday before, which is known; from Tuesday on it is an hour of the week itself, which the forecast is for. Add it, and the backtest says 38.7 instead of 42.0: better, and false. See *Only lags the horizon allows* in *The idea*.",
            "temp": "The temperature as it turned out is known only afterwards. With the weather as it turned out, the backtest says 34.2: about what a perfect weather forecast gives here. A real forecast can only use a weather forecast made at the origin. See step 5 of the slider.",
            "right": "Weekends and holidays are known long before: `workingday` is a calendar feature. On the 11 holidays of 2012, the tree with the hour, the weekday and the month still misses by 93.9 riders an hour; with `workingday` too, by 69.8.",
        },
    )
    choice_known
    return (choice_known,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Lag features put the autocorrelation of [Does today look like yesterday?](/course/open/ts-autocorrelation?lang=en) to work: the lags that correlate most are the columns worth having. The table, the label and the tree are those of [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en); the backtest and the seasonal naive forecast are those of [Testing a forecast on the past](/course/open/ts-backtesting?lang=en), and lags of at least the horizon keep it honest. This ends the module on time series. The next module is about recommending, with a table that is almost empty: which customer wants which product.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_known, result_lag_features, result_tree_forecast):
    L.wrap_up(result_lag_features, result_tree_forecast, choice_known)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
