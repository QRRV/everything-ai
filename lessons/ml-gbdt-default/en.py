# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-gbdt-default. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-gbdt-default", "en")
    C = L.checks   # the clean table of traffic, the test MAE of every number of bins on the slider, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Tree after tree on what is left: gradient boosting](/course/open/ml-gradient-boosting?lang=en) and one about [Forecasting demand with trees](/course/open/ts-forecast-demand?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_trees():
    quick_trees = L.choice(
        "quick-trees",
        question="On the penguins, the validation RMSE of gradient boosting was smallest at 42 trees, 415 g, and rose to 575 g at 1,000 trees. How do you choose the number of trees?",
        options={
            "right": "Where the error on the validation set is smallest",
            "more": "As many as the computer allows: every tree learns more",
            "train": "Where the error on the training set is smallest",
        },
        correct="right",
        why={
            "more": "More trees keep lowering the training error, but after a while they learn the quirks of single training examples: on the penguins the validation RMSE rose from 415 g to 575 g. See step 5 of the recipe in [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en).",
            "train": "The training error falls with every tree, to 158 g at 1,000 trees, so it would always choose the most. See step 5 of the recipe in [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en).",
            "right": "Today the model does that by itself while it learns.",
        },
    )
    quick_trees
    return (quick_trees,)


@app.cell(hide_code=True)
def quick_check_calendar():
    quick_calendar = L.choice(
        "quick-calendar",
        question="A table of traffic, one row per hour. Which of these columns is a calendar feature?",
        options={
            "right": "`weekday`: the day of the week of every hour",
            "lag": "`lag_1`: the traffic of the hour before",
            "temp": "`temp`: the temperature of every hour",
        },
        correct="right",
        why={
            "lag": "That is a lag feature: the series itself, one step earlier. A calendar feature is what a date says about itself. See *Calendar features* in *The idea* in [Forecasting demand with trees](/course/open/ts-forecast-demand?lang=en).",
            "temp": "The temperature is measured in that hour; the date alone does not tell it. A calendar feature is what a date says about itself. See *Calendar features* in *The idea* in [Forecasting demand with trees](/course/open/ts-forecast-demand?lang=en).",
            "right": "Today the hour, the weekday and the month are features again, and you make one more calendar feature yourself: the holidays.",
        },
    )
    quick_calendar
    return (quick_calendar,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of the data of companies and institutions is a table: orders, customers, machines, patients, one row per thing, numbers and words mixed, some cells empty. On such tables one kind of model is usually the first to try: gradient-boosted trees. In a study from 2022 on 45 tables of medium size, trees, gradient boosting among them, did better than the neural networks that were tried. Of the 29 winning entries that the competition site Kaggle described on its blog in 2015, 17 used XGBoost, a library for gradient boosting; LightGBM and CatBoost are two others. The version of scikit-learn, `HistGradientBoostingRegressor`, is modeled on LightGBM, and you met it in [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en). Today you see why it is the default: what it does with numbers, with words, with empty cells and with the number of trees, so that its default settings already do well.

    **An everyday example.** A café wants to know when it is busy. It sorts its receipts into 24 piles, one per hour, not by the exact minute: 24 piles show everything that matters. With 6 piles of 4 hours, the quiet early morning and the start of the morning rush end up on one pile, and the café can no longer tell them apart.
    """)
    return


@app.cell(hide_code=True)
def predict_boost_cell():
    guess_boost = L.guess(
        "boost-guess",
        question="Traffic on a highway, hour by hour: learn from the hours before 2017, test on the 15,246 hours of 2017 and 2018. A random forest of 100 trees misses by 274.8 vehicles an hour on average. Gradient boosting with its default settings learns from the same table. What does it miss by, and how long does it take?",
        options={
            "slow": "About 275, and it takes longer",
            "right": "About 260, in a twentieth of the time",
            "half": "About 140, half, in the same time",
        },
        reveal="**260.3, in about a twentieth of the time.** In a test of the course, on one core, the forest took 6.7 seconds and gradient boosting 0.3. A linear regression misses by 570.4, and always saying the mean of the training hours by 1,738.3. Gradient boosting is not far ahead of the forest here. What makes it the default is that it does this well with its default settings, and fast.",
    )
    guess_boost
    return (guess_boost,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The table.** The traffic on Interstate 94 of [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en), cleaned as there: no hours at 0 K, not the hour with 9,831.3 mm of rain, and one row per hour. That leaves 40,564 hours from October 2012 to September 2018. The label is `traffic_volume`, the vehicles counted in that hour. The features are the calendar features of [Forecasting demand with trees](/course/open/ts-forecast-demand?lang=en), the hour, the weekday and the month; the weather as numbers, the temperature in °C, rain and snow in mm and the clouds in %; and the weather in a word, such as `Clear`, `Rain` or `Snow`, 11 words in all. The models learn from the 25,318 hours before 2017 and are tested on the 15,246 hours of 2017 and 2018: on later hours, as in [Testing a forecast on the past](/course/open/ts-backtesting?lang=en), because the model will be used on hours that come later.

        **Bins.** Before it grows a single tree, `HistGradientBoostingRegressor` sorts the values of every feature into at most `max_bins` **bins**, 255 by default: each bin a range of values, with about as many rows in each. A feature with at most that many different values gets a bin per value: `hour` has 24. A tree then only asks about the borders between bins. To find the best question about a feature, it adds up the residuals of the rows in every bin, a histogram, and tries only the borders: far less work than trying every value of every row. That is the *histogram* in **histogram gradient boosting**. Values in one bin look the same to every tree.

        **Empty cells.** A missing value gets a bin of its own. At every question, the tree tries the rows of that bin on both sides and sends them where they help most. That is how it handles missing values by itself, as [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en) said. The clean traffic table has no empty cells; your own tables will.

        **Words.** A linear model needs numbers. For the weather in a word, every word gets a column that is True in the hours with that word and False in the others: the one-hot vectors of [One-hot vectors for classes](/course/open/ml-one-hot?lang=en) as columns: `pd.get_dummies` makes them. A linear model needs the same for the hour, the weekday and the month, because the traffic does not rise in a straight line with the hour: the 8 features become 58 columns. A tree needs no such columns for the hour: it can ask whether the hour is at most 6, and then whether it is at most 9. And for words, `HistGradientBoostingRegressor` needs no one-hot columns either. A column of the pandas type `category`, made with `astype("category")`, it takes as it is, and at a question it can send any group of words to one side, such as `Rain`, `Snow` and `Thunderstorm` against the rest. That is **native categorical support**. On this table the word hardly matters: with the weather in a word of type `category`, the test MAE is 260.3, with 11 one-hot columns 259.1, and without the word at all 259.8. The weather as numbers and the calendar already tell almost everything.

        **Early stopping.** In [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en) you chose the number of trees on the validation set: the error there falls, then rises again. The model can do that itself while it learns: after every tree it measures the error on a validation set, and stops once 10 trees in a row did not lower it. That is **early stopping**. `early_stopping=True` turns it on, and `fit(X, y, X_val=..., y_val=...)` gives it the validation set; `n_iter_` then says how many trees the model kept. The error it watches is the squared error it learns on: the RMSE on the validation set.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **histogram gradient boosting** | gradient boosting that first sorts every feature into bins and only asks about the borders between them; fast on large tables | `HistGradientBoostingRegressor()` |
        | **bin** | a range of values of a feature; every tree treats the values in one bin the same | `max_bins=255` |
        | **native categorical support** | the model takes a column of words as it is: a question can send any group of words to one side | `df["weather_main"].astype("category")` |
        | **early stopping** | stop adding trees once the error on a validation set has not gone down for a number of trees in a row | `early_stopping=True` |
        | `X_val`, `y_val` | the validation set that `fit` stops on | `model.fit(X_train, y_train, X_val=X_val, y_val=y_val)` |
        | `n_iter_` | how many trees the model kept | `model.n_iter_` |
        | `pd.get_dummies` | the table with a column of True and False for every value of the columns named | `pd.get_dummies(X, columns=["weather"])` |
        | `notna()` | True where a value is there: the opposite of `isna()` | `df["holiday"].notna()` |
        | `index.date` | the date of every moment of a datetime index, without its time | `df.index.date` |
        """),
        L.watch_out("With its default, `early_stopping=\"auto\"`, the model stops early only when it learns from more than 10,000 rows, and then on a tenth of those rows picked at random, which it no longer learns from. The worked example has 25,318 training hours, so it happens there: the model still keeps all 100 trees it grows by default, and learns from nine tenths of the hours. With `early_stopping=False` it learns from all of them and misses by 258.1 instead of 260.3. On a table in time, a random tenth also lies between the hours it learns from. Choose yourself: `early_stopping=True` with a validation set after the training hours, or `False`."),
        mo.md(r"""
        The slider sets `max_bins`. The gray bars are the mean traffic of every hour of the day in the training hours. The <span class="c-param">orange</span> lines are what a tree that asks only about `hour` can tell apart: one value per bin, the mean of its hours. The dashed lines are the borders between bins. Under the picture, the test MAE of the model with that many bins, measured beforehand by the course. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_bins_cell():
    def _frame(bins):
        said = C.bins_said(bins, L.lang)
        alt = ("A bar chart of the mean traffic of every hour of the day in the training hours, from under 400 vehicles at "
               "night to about 5,600 at 16:00, with a horizontal line for every bin of the hour at the mean of its hours. " + said)
        return (C.bins_picture(bins, {"title": "Traffic per hour of the day", "hour": "hour of the day",
                                      "vehicles": "vehicles an hour", "alt": alt}), said)

    explore_bins = L.explore("max-bins", "max_bins", _frame, start=255, values=C.BINS, role="hyper",
                             task="Go from 255 bins down to 12. From how many bins on does every hour have a bin of its own, and what happens to the MAE below that?")
    explore_bins
    return


@app.cell(hide_code=True)
def text_after_bins():
    mo.md(r"""
    With 12 bins, `hour` gets 12 bins, most of two hours. Hour 4, with 684 vehicles on average, shares a bin with hour 5, with 2,057: the start of the morning rush. No tree can tell them apart, and the test MAE is 395.6. That is the hour: with only `hour` cut into those 12 groups and 255 bins for the other features, it is 397.3, about the same. With 16 bins the MAE is 334.4, with 20 bins 304.1. From 24 bins on, every hour has a bin of its own, and the MAE stays between 258.9 and 260.6. The default of 255 is plenty here.

    ### Recipe

    To start on a table:

    1. Make one row per thing, with the label and features that are known when you predict.
    2. Make word columns of type `category` with `astype("category")`, and leave empty cells empty.
    3. Split off a test part, by time when the rows lie in time, and a validation set to stop on.
    4. Fit `HistGradientBoostingRegressor`, or `HistGradientBoostingClassifier` for classes, with its default settings, `early_stopping=True` and the validation set.
    5. Compare with a baseline, such as the mean, and with a simple model: the trees have to beat both clearly.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes the clean table and the features, splits at 1 January 2017, and compares four models on the test hours: always the mean of the training hours, as a baseline; a linear regression on 58 columns, with one-hot columns for the calendar and the word; a random forest of 100 trees, with one-hot columns for the word, because the forest of scikit-learn takes no words; and histogram gradient boosting with its default settings, on the table as it is. `threadpool_limits(1)` lets scikit-learn calculate on one core, as in [Forecasting demand with trees](/course/open/ts-forecast-demand?lang=en). The forest takes several seconds: in a test of the course, on one core, 6.7 seconds, against 0.3 for gradient boosting. The cell prints the number of hours, training hours and test hours, the number of columns of the linear regression, and the four MAEs. Run the cell.
    """)
    return


@app.cell
def example_compare():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from threadpoolctl import threadpool_limits

    # 1. the clean table of the project: one row per hour, the time as the index
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    _df = _df[_df["temp"] > 0]
    _df = _df[_df["rain_1h"] < 100]
    _df = _df.drop_duplicates(subset="date_time")
    _df["date_time"] = pd.to_datetime(_df["date_time"])
    _df = _df.set_index("date_time")

    # the features: the calendar, the weather as numbers, the weather in a word of type category
    _X = pd.DataFrame(index=_df.index)
    _X["hour"] = _df.index.hour
    _X["weekday"] = _df.index.dayofweek
    _X["month"] = _df.index.month
    _X["temp"] = _df["temp"] - 273.15
    _X["rain"] = _df["rain_1h"]
    _X["snow"] = _df["snow_1h"]
    _X["clouds"] = _df["clouds_all"]
    _X["weather"] = _df["weather_main"].astype("category")   # 2. words of type category
    _y = _df["traffic_volume"]

    # 3. learn from the hours before 2017, test on 2017 and 2018
    _train = _X.index < "2017-01-01"
    _test = _X.index >= "2017-01-01"
    print(len(_X), _train.sum(), _test.sum())

    # 5. a baseline: always the mean of the training hours
    _mean = np.mean(abs(_y[_test] - _y[_train].mean()))

    # a linear regression needs numbers: one-hot columns for the calendar and the word
    _D = pd.get_dummies(_X, columns=["hour", "weekday", "month", "weather"])
    print(_D.shape[1])
    _linear = LinearRegression().fit(_D[_train], _y[_train])
    _lin = np.mean(abs(_y[_test] - _linear.predict(_D[_test])))

    with threadpool_limits(1):
        # the forest of scikit-learn takes no words: one-hot columns for the word
        _F = pd.get_dummies(_X, columns=["weather"])
        _forest = RandomForestRegressor(random_state=0).fit(_F[_train], _y[_train])
        _rf = np.mean(abs(_y[_test] - _forest.predict(_F[_test])))
        # 4. gradient boosting takes the table as it is
        _boost = HistGradientBoostingRegressor(random_state=0).fit(_X[_train], _y[_train])
        _gb = np.mean(abs(_y[_test] - _boost.predict(_X[_test])))
    print(round(_mean, 1), round(_lin, 1), round(_rf, 1), round(_gb, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The table has 40,564 hours: 25,318 to learn from and 15,246 to test on. The linear regression gets 58 columns: the 4 numbers of the weather, 24 for the hour, 7 for the weekday, 12 for the month and 11 for the word. Always the mean misses by 1,738.3 vehicles an hour, the linear regression by 570.4, the forest by 274.8 and gradient boosting by 260.3. Even with a column for every hour, the linear regression adds one number for the hour and one for the weekday, so the difference between a Monday and a Sunday comes out the same at every hour. In the training hours it is not: at 7:00 a Monday has 4,655 vehicles more than a Sunday on average, at 2:00 it has 343 fewer. A tree asks about both, one after the other.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_whole_day():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A holiday for the whole day

        The column `holiday` has the name of a holiday, such as `Independence Day`, but only in the first hour of that day, as you saw in the project; every other hour has NaN. A model that gets that column sees a holiday only at midnight. `whole_day(df)` should give a list with a mark for every row of `df`: 1 when the day of the row is a holiday, that is, when one of the rows of that day has a name, else 0. `days = df.index.date` gives the date of every row. Replace the two `...`. The check calls `whole_day(traffic)`, with `traffic` the clean table with its column `holiday`, and tries four made-up tables, one with missing hours.
        """),
        L.task("Let `whole_day` mark every hour of a day that has a holiday name in one of its hours."),
    ])
    return


@app.cell(hide_code=True)
def exercise_whole_day():
    box_whole_day = L.code_box("whole-day", lines=9)
    box_whole_day
    return (box_whole_day,)


@app.cell(hide_code=True)
def check_whole_day(box_whole_day):
    result_whole_day = L.check("whole-day", box_whole_day.value)
    result_whole_day
    return (result_whole_day,)


@app.cell(hide_code=True)
def hints_whole_day():
    L.hints(
        "whole-day",
        [
            "Two steps: first the dates that have a name in one of their hours, then for every row whether its date is one of them.",
            "The hours with a name are where `holiday` is not missing: the opposite of `isna()`. An array with a mask between its brackets keeps the values where the mask is True, as in [Selecting parts of an array](/course/open/np-indexing?lang=en), and `set` keeps each date once. `in` asks whether a value is in a set.",
            "`holidays = set(days[df[\"holiday\"].___()])` and `if day ___ holidays:`.",
        ],
        why="The 53 names become 1,203 marked hours. With that column the test MAE drops from 260.3 to 235.8 vehicles an hour, and on the 18 holidays of 2017 and 2018 from 913.5 to 507.3: the model now sees a holiday in every hour of it, not only at midnight.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stop_on():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Stop on 2016

        `stop_on(X_train, y_train, X_val, y_val)` should fit `HistGradientBoostingRegressor` with at most 300 trees and early stopping on the validation set, and give back how many trees it kept. Replace the two `...`. The check calls `stop_on(X_train, y_train, X_val, y_val)`: the features of the worked example with the column `holiday` of exercise 1, the 17,481 hours before 2016 to learn from and the 7,837 hours of 2016 to stop on. It also tries two made-up tables of 2,000 and 5,000 rows.
        """),
        L.task("Let `stop_on` stop early on the validation set and give the number of trees it kept."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stop_on():
    box_stop_on = L.code_box("stop-on", lines=7)
    box_stop_on
    return (box_stop_on,)


@app.cell(hide_code=True)
def check_stop_on(box_stop_on):
    result_stop_on = L.check("stop-on", box_stop_on.value)
    result_stop_on
    return (result_stop_on,)


@app.cell(hide_code=True)
def hints_stop_on():
    L.hints(
        "stop-on",
        [
            "Two things: turn early stopping on in the model, and give `fit` the validation set besides the rows it learns from.",
            "`early_stopping` takes `True`, `False` or `\"auto\"`, and in scikit-learn 1.9 `\"auto\"` turns it on only above 10,000 rows, even when `X_val` is given. `fit` takes the validation set as two keyword arguments after the features and the label.",
            "`early_stopping=___` and `model.fit(X_train, y_train, X_val=___, y_val=___)`.",
        ],
        why="The model stops at 50 trees: the error on 2016 was lowest after 40, and the 10 trees after that did not get it lower. On 2017 and 2018 the 50 trees miss by 245.7 vehicles an hour, and all 300 trees by 243.8: about the same, with a sixth of the trees. Both are above the 235.8 of exercise 1, because these models learned only from the hours before 2016.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Exercise 3. The first model on a new table
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-model",
        question="A new table: 50,000 members of a gym, 30 columns of numbers and words, some cells empty, and the question who cancels next month. Which model do you try first?",
        options={
            "right": "`HistGradientBoostingClassifier` with its default settings, the words as `category`, early stopping on a validation set",
            "linear": "A logistic regression, after one-hot columns for every word and a value in every empty cell",
            "tune": "Gradient boosting, but only after a long search over all its settings",
        },
        correct="right",
        why={
            "linear": "It can work, and it is a good model to compare with. But it needs that work first, and it only adds up effects: on the traffic, a linear regression on 58 columns missed by 570.4 vehicles an hour, gradient boosting on the table as it was by 260.3.",
            "tune": "The default settings already do well: on the traffic, gradient boosting beat the forest without a single setting changed. Search the settings once you have a first model to beat; *Searching settings: grid search and random search* shows how.",
            "right": "The first model to try on a table; `HistGradientBoostingClassifier` is the same model for classes. Compare it with a baseline, such as always the most common class, before you trust its score.",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The trees are those of [Tree after tree on what is left](/course/open/ml-gradient-boosting?lang=en); early stopping chooses their number on a validation set, as you did there by hand, and in *Early stopping* you will do the same for a neural network, epoch by epoch. The calendar features come from [Forecasting demand with trees](/course/open/ts-forecast-demand?lang=en), and the clean table from [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en). In the next lesson, *What competitions teach: the leaderboard*, you see what happens when you keep choosing on the same part of the data.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_stop_on, result_whole_day):
    L.wrap_up(result_whole_day, result_stop_on, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
